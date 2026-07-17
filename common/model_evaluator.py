from typing import cast, TypedDict

import os
from datetime import datetime
import json

import torch
from datasets import DatasetDict
from sacrebleu import corpus_bleu, corpus_chrf
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from torch.utils.data import DataLoader

from common.model_trainer import ModelTrainer
from common.process_word_windows import encode_text
from common.utils import decode_fst_output, get_device, get_lang_abbrev, qs_tokenized_dataloader, \
    TokenizedBatch, SPANISH_LANG_ID, QUECHUA_LANG_ID

class TranslationTrainingConfig(TypedDict):
    epochs: int
    batch_size: int
    batches_per_update: int
    lr: float
    weight_decay: float
    warmup_steps_frac: float
    grad_clip_max_norm: float
    eval_freq: int    
    save_folder_name: str | None

class TranslationMetrics(TypedDict):
    bleu_base: float
    chrf_base: float
    chrf_pp_base: float
    bleu_fst: float
    chrf_fst: float
    chrf_pp_fst: float

class TranslationTrainingResult(TypedDict):
    train_losses: list[float]
    val_losses: list[float]
    val_metrics: list[TranslationMetrics]

class TranslationEvaluator():
    '''Evaluates the finetuning of FST models across different hyperparameters and records the results.'''
    _CHECKPOINT_STR_START = 'translation_checkpoint'
    _TRAINING_RESULTS_FILENAME = 'training_result.json'
    _MAX_LENGTH = 128
    _MAX_LENGTH_RESPONSE = 160
    _NUM_BEAMS = 1
    _N_DATALOADER_WORKERS = 1
    _N_TOKENIZE_WORKERS = 4
    _TRANSLATION_BATCHES_PER_PRINT = 1000

    def __init__(
            self,
            model: M2M100ForConditionalGeneration,
            tokenizer: NllbTokenizer,
            dataset_dict: DatasetDict,
            old_vocab_size: int,
            device: str | None = None,
            use_fst: bool = True
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_dict = dataset_dict
        self.old_vocab_size = old_vocab_size
        self.device = get_device() if device is None else device
        # When False, Quechua is tokenized with NLLB's default subword tokenizer
        # instead of FST morpheme segmentation (used for the reference/control model).
        self.use_fst = use_fst

        # Gradient checkpointing disabled: at batch_size=4 there's ample VRAM on a 24GB GPU, and it
        # only trades ~20-30% speed for memory. Gradients (and thus the trained model) are identical.
        # self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

    def train_model(
            self,
            config: TranslationTrainingConfig
    ) -> TranslationTrainingResult:
        '''
        Finetune the model using the given config. This function returns training losses, validation losses, and translation
        metrics (BLEU, chrF, chrF++) on the validation set every config['eval_freq'] epochs. This function also saves
        checkpoints to config['save_folder_name'] if it is provided.
        '''
        assert config['epochs'] > 0
        assert config['batch_size'] > 0
        assert config['batches_per_update'] > 0
        assert config['lr'] > 0
        assert config['weight_decay'] >= 0
        assert 0 <= config['warmup_steps_frac'] <= 1
        assert config['grad_clip_max_norm'] > 0
        assert config['eval_freq'] > 0

        train_loader = self._build_dataloader('train', config['batch_size'], shuffle=True)
        val_loader = self._build_dataloader('validation', config['batch_size'], shuffle=False)

        trainer = ModelTrainer(
            modified_model=self.model,
            modified_tokenizer=self.tokenizer,
            n_training_epochs=config['epochs'],
            n_batches_train_dataset=len(train_loader),
            batches_per_update=config['batches_per_update'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            warmup_steps_frac=config['warmup_steps_frac'],
            grad_clip_max_norm=config['grad_clip_max_norm'],
            device=self.device
        )
        trainer.freeze_old_embeddings(self.old_vocab_size)
        trainer.freeze_encoder()

        results = self._run_training(trainer, train_loader, val_loader, config)

        if config['save_folder_name'] is not None:
            self._save_training_results(results, config['save_folder_name'])

        return results

    def eval_model(
            self,
            batch_size: int,
            split: str = 'test',
            use_decoded_fst_output: bool = False,
            log_freq: int = _TRANSLATION_BATCHES_PER_PRINT
    ) -> TranslationMetrics:
        '''Evaluates the model on the given dataset split with BLEU, chrF, and chrF++ translation metrics.'''
        loader = self._build_dataloader(split, batch_size=batch_size, shuffle=False)
        metrics = self._compute_translation_metrics(loader, split, use_decoded_fst_output, log_freq)
        self._print_translation_metrics(metrics)
        return metrics

    def _run_training(
            self,
            trainer: ModelTrainer,
            train_loader: DataLoader[TokenizedBatch],
            val_loader: DataLoader[TokenizedBatch],
            config: TranslationTrainingConfig
    ) -> TranslationTrainingResult:
        train_losses: list[float] = []
        val_losses: list[float] = []
        val_metrics: list[TranslationMetrics] = []

        start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        for epoch in range(1, config['epochs'] + 1):
            print(f"--------------- Epoch {epoch}/{config['epochs']} ---------------")
            print(f'Starting training epoch...')
            train_loss = trainer.train_epoch(train_loader, config['batches_per_update'])

            print(f'Starting validation epoch...')
            val_loss = trainer.eval_epoch(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch} - train loss: {train_loss:.5f}, val loss: {val_loss:.5f}")

            if epoch % config['eval_freq'] == 0:
                print(f'Computing translation metrics on the validation dataset...')
                metrics = self._compute_translation_metrics(val_loader, 'validation')
                val_metrics.append(metrics)
                self._print_translation_metrics(metrics)

            if config['save_folder_name'] is not None:
                self._save_checkpoint(config['save_folder_name'], epoch, start_time)

        return self._format_result(train_losses, val_losses, val_metrics)

    def _compute_translation_metrics(
            self,
            data_loader: DataLoader[TokenizedBatch],
            split: str,
            use_decode_fst_output: bool = False,
            log_freq: int = _TRANSLATION_BATCHES_PER_PRINT
    ) -> TranslationMetrics:
        dataset = self.dataset_dict[split]
        predicted_translations: list[str] = []

        bos_target_lang = cast(int, self.tokenizer.convert_tokens_to_ids(QUECHUA_LANG_ID))

        self.model.eval()
        n_batches = len(data_loader)

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch: TokenizedBatch
                if (i + 1) % log_freq == 0:
                    print(f'translating batch {i + 1}/{n_batches}')

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                output = cast(torch.LongTensor, self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    forced_bos_token_id=bos_target_lang,
                    max_length=TranslationEvaluator._MAX_LENGTH_RESPONSE,
                    num_beams=TranslationEvaluator._NUM_BEAMS
                ))

                decoded_batch = self.tokenizer.batch_decode(
                    output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                if use_decode_fst_output:
                    decoded_batch = map(decode_fst_output, decoded_batch)

                predicted_translations.extend(decoded_batch)

        base_reference_translations: list[str] = dataset[get_lang_abbrev(QUECHUA_LANG_ID)]

        bleu_base = corpus_bleu(predicted_translations, [base_reference_translations])
        chrf_base = corpus_chrf(predicted_translations, [base_reference_translations])
        chrf_pp_base = corpus_chrf(predicted_translations, [base_reference_translations], word_order=2)

        if not self.use_fst:
            # The standard-tokenization model produces natural Quechua, so there is no
            # FST-encoded reference to score against; mirror the base metrics.
            return TranslationMetrics({
                'bleu_base': bleu_base.score,
                'chrf_base': chrf_base.score,
                'chrf_pp_base': chrf_pp_base.score,
                'bleu_fst': bleu_base.score,
                'chrf_fst': chrf_base.score,
                'chrf_pp_fst': chrf_pp_base.score,
            })

        def identity(x: str): return x
        map_fn = decode_fst_output if use_decode_fst_output else identity

        fst_reference_translations: list[str] = [map_fn(self._encode_reference(brt)) for brt in base_reference_translations]

        bleu_fst = corpus_bleu(predicted_translations, [fst_reference_translations])
        chrf_fst = corpus_chrf(predicted_translations, [fst_reference_translations])
        chrf_pp_fst = corpus_chrf(predicted_translations, [fst_reference_translations], word_order=2)

        return TranslationMetrics({
            'bleu_base': bleu_base.score,
            'chrf_base': chrf_base.score,
            'chrf_pp_base': chrf_pp_base.score,
            'bleu_fst': bleu_fst.score,
            'chrf_fst': chrf_fst.score,
            'chrf_pp_fst': chrf_pp_fst.score,
        })

    def _encode_reference(self, text: str) -> str:
        '''
        Processes text through encode_text and the tokenizer so the reference matches the
        form it is trained to produce (including tags for morpheme markers).
        '''
        chunks = encode_text(text, self.tokenizer)
        token_ids = cast(list[int], self.tokenizer(
            chunks,
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=True,
            max_length=TranslationEvaluator._MAX_LENGTH,
        )['input_ids'])

        return cast(str, self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ))

    def _build_dataloader(
            self,
            split: str,
            batch_size: int,
            shuffle: bool
    ) -> DataLoader[TokenizedBatch]:
        return qs_tokenized_dataloader(
            self.dataset_dict[split],
            self.tokenizer,
            SPANISH_LANG_ID,
            batch_size,
            max_length=TranslationEvaluator._MAX_LENGTH,
            shuffle=shuffle,
            n_dataloader_workers=TranslationEvaluator._N_DATALOADER_WORKERS,
            n_tokenize_workers=TranslationEvaluator._N_TOKENIZE_WORKERS,
            use_fst=self.use_fst
        )

    def _format_result(
            self,
            train_losses: list[float],
            val_losses: list[float],
            val_metrics: list[TranslationMetrics]
    ) -> TranslationTrainingResult:
        return TranslationTrainingResult({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics': val_metrics
        })

    def _save_checkpoint(self, save_folder_name: str, epoch: int, start_time: str) -> None:
        checkpoint_name = f'{TranslationEvaluator._CHECKPOINT_STR_START}_epoch{epoch}_{start_time}'
        full_path = os.path.join(save_folder_name, checkpoint_name)
        self.model.save_pretrained(full_path)
        self.tokenizer.save_pretrained(full_path)

        existing = sorted(
            [d for d in os.listdir(save_folder_name) if d.startswith(TranslationEvaluator._CHECKPOINT_STR_START)],
            key=lambda d: os.path.getmtime(os.path.join(save_folder_name, d))
        )
        for old in existing[:-2]:
            old_path = os.path.join(save_folder_name, old)
            for f in os.listdir(old_path):
                os.remove(os.path.join(old_path, f))
            os.rmdir(old_path)

    def _save_training_results(self, results: TranslationTrainingResult, save_folder_name: str) -> None:
        os.makedirs(save_folder_name, exist_ok=True)
        full_path = os.path.join(save_folder_name, TranslationEvaluator._TRAINING_RESULTS_FILENAME)
        with open(full_path, 'w') as f:
            json.dump(results, f, indent=2)

    
    def _print_translation_metrics(self, metrics: TranslationMetrics) -> None:
        print(
            f"\t- BLEU (base):   {metrics['bleu_base']:.3f}\n"
            f"\t- chrF (base):   {metrics['chrf_base']:.3f}\n"
            f"\t- chrF++ (base): {metrics['chrf_pp_base']:.3f}\n"
            f"\t- BLEU (fst):   {metrics['bleu_fst']:.3f}\n"
            f"\t- chrF (fst):   {metrics['chrf_fst']:.3f}\n"
            f"\t- chrF++ (fst): {metrics['chrf_pp_fst']:.3f}"
        )
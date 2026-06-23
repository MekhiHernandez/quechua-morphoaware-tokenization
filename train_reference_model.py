from datasets import load_dataset

from transformers import NllbTokenizer

from common.utils import get_device, load_model
from common.model_evaluator import TranslationEvaluator, TranslationTrainingConfig

def _get_vocab_size(model_path: str) -> int:
    return len(NllbTokenizer.from_pretrained(model_path))

'''
Trains the standard-tokenization reference (control) model for the FST experiment.

This finetunes the base NLLB model on the same dataset with the same recipe as
train_fst_model.py (frozen encoder, frozen pre-existing embeddings, decoder trainable),
but using NLLB's default subword tokenizer for Quechua instead of FST morpheme
segmentation. It is the apples-to-apples baseline that isolates the effect of
morpheme-aware tokenization: same model, same data, same trainable parameters, only
the tokenization differs.

Since there is no vocabulary extension here, every token is a "pre-existing" token, so
passing old_vocab_size = len(base tokenizer) makes freeze_old_embeddings freeze the
entire embedding matrix - mirroring the FST recipe, which trains only the new morpheme
embeddings plus the decoder.
'''

quechua_spanish_dataset_id = 'somosnlp-hackathon-2022/spanish-to-quechua'
base_model_load_path = 'nllb-model'
model_save_path = './nllb-model-standard-trained'

if __name__ == '__main__':
    device = get_device()
    tokenizer, model = load_model(device, base_model_load_path)
    dataset_dict = load_dataset(quechua_spanish_dataset_id)

    old_vocab_size = _get_vocab_size(base_model_load_path)

    evaluator = TranslationEvaluator(
        model=model,
        tokenizer=tokenizer,
        dataset_dict=dataset_dict,
        old_vocab_size=old_vocab_size,
        device=device,
        use_fst=False,
    )

    config: TranslationTrainingConfig = {
        'epochs': 20,
        'batch_size': 4,
        'batches_per_update': 16,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps_frac': 0.1,
        'grad_clip_max_norm': 1.0,
        'eval_freq': 3,
        'save_folder_name': model_save_path,
    }

    result = evaluator.train_model(config)

    tokenizer.save_pretrained(model_save_path)
    model.save_pretrained(model_save_path)

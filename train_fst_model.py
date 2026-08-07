from datasets import load_dataset

from transformers import NllbTokenizer

from common.utils import get_device, load_model
from common.model_evaluator import TranslationEvaluator, TranslationTrainingConfig

def _get_vocab_size(model_path: str) -> int:
    return len(NllbTokenizer.from_pretrained(model_path))

'''
This script runs the training process with our fst morpheme aware model. Run extend_model_vocabulary.py
before this. On my machine, running the training process with these parameters uses just under 8Gbs of VRAM.
'''

quechua_spanish_dataset_id = 'somosnlp-hackathon-2022/spanish-to-quechua'
base_model_load_path = 'nllb-model'
model_load_path = './nllb-model-fst'
model_save_path = './nllb-model-fst-trained'

if __name__ == '__main__':
    device = get_device()
    tokenizer, model = load_model(device, model_load_path)
    dataset_dict = load_dataset(quechua_spanish_dataset_id)

    old_vocab_size = _get_vocab_size(base_model_load_path)

    evaluator = TranslationEvaluator(
        model=model,
        tokenizer=tokenizer,
        dataset_dict=dataset_dict,
        old_vocab_size=old_vocab_size,
        device=device,
    )

    config: TranslationTrainingConfig = {
        'epochs': 60,                      # a CAP; early stopping ends it once val loss plateaus
        'batch_size': 4,
        'batches_per_update': 16,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps_frac': 0.1,
        'grad_clip_max_norm': 1.0,
        'eval_freq': 3,
        'save_folder_name': model_save_path,
        'early_stopping_patience': 5,      # stop after 5 epochs with no val-loss improvement
        'early_stopping_min_delta': 0.0,
    }

    result = evaluator.train_model(config)

    tokenizer.save_pretrained(model_save_path)
    model.save_pretrained(model_save_path)
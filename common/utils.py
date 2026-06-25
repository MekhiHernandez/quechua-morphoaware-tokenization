from typing import Any, cast, Literal, TypedDict

from datasets import Dataset

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, M2M100ForConditionalGeneration, NllbTokenizer

from common.process_word_windows import encode_text

QUECHUA_LANG_ID: Literal['quy_Latn'] = 'quy_Latn'
SPANISH_LANG_ID: Literal['spa_Latn'] = 'spa_Latn'

def decode_fst_output(text: str) -> str:
    '''Converts FST-segmented model output back into natural Quechua words.'''
    return ' '.join(text.replace('=', ' ').replace('+', '').replace(' ⩲', ' ').replace('⩲', '').split())

class TokenizedBatch(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_lang_abbrev(lang: Literal['spa_Latn', 'quy_Latn']) -> Literal['es', 'qu']:
    if lang == SPANISH_LANG_ID:
        return 'es'
    return 'qu'

def get_other_lang(lang: Literal['spa_Latn', 'quy_Latn']) -> Literal['spa_Latn', 'quy_Latn']:
    return QUECHUA_LANG_ID if lang == SPANISH_LANG_ID else SPANISH_LANG_ID

def load_model(device: str, load_path: str = 'nllb-model') -> tuple[NllbTokenizer, M2M100ForConditionalGeneration]:
    tokenizer = NllbTokenizer.from_pretrained(load_path)
    model = M2M100ForConditionalGeneration.from_pretrained(load_path, device_map=device)
    return tokenizer, model

def save_model(tokenizer: NllbTokenizer, model: M2M100ForConditionalGeneration, save_path: str) -> None:
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

def qs_tokenized_dataloader(
        dataset: Dataset,
        tokenizer: NllbTokenizer,
        source_lang: Literal['spa_Latn', 'quy_Latn'],
        batch_size: int,
        max_length: int = 512,
        shuffle: bool = True,
        n_dataloader_workers: int = 0,
        n_tokenize_workers: int = 0,
        use_fst: bool = False
) -> DataLoader[TokenizedBatch]:
    '''
    Returns a dataloader that's pre-tokenized.
    Args:
        dataset: dataset to tokenize
        tokenizer: tokenizer used to tokenize dataset
        source_lang: the source language, either Spanish (spa_Latn) or Quechua (quy_Latn)
        batch_size: batch size when iterating over dataloader
        max_length: determines the length that inputs are truncated to
        shuffle: whether or not to shuffle the dataloader
        n_dataloader_workers: number processes to load data in parallel for dataloader
        n_tokenize_workers: number of processes to help with pre-tokenization
        use_fst: If true use run_fst to tokenize Quechua, else use nllb tokenizer
    '''
    assert('qu' in dataset.column_names and 'es' in dataset.column_names)
    qs_token_fn = _qs_tokenize_fst_fn if use_fst else _qs_tokenize_fn

    tokenized = dataset.map(
        qs_token_fn(tokenizer, source_lang, max_length),
        batched=True,
        num_proc=n_tokenize_workers
    )
    tokenized = tokenized.select_columns(['input_ids', 'attention_mask', 'labels'])

    collector = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        return_tensors='pt'
    )

    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = get_other_lang(source_lang)

    return DataLoader(
        cast(torch.utils.data.Dataset[TokenizedBatch], tokenized),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_dataloader_workers,
        persistent_workers=n_dataloader_workers > 0,
        collate_fn=collector
    )

def _qs_tokenize_fn(
        tokenizer: NllbTokenizer,
        source_lang: Literal['spa_Latn', 'quy_Latn'],
        max_length: int = 512
):
    target_lang = get_other_lang(source_lang)
    def tokenize_helper(batch: Any):
        # Set languages here (not in qs_tokenized_dataloader, which sets them only after
        # .map() has already run) so the NLLB language tokens prepended to the inputs and
        # labels are correct. Without this the tokenizer's default (eng_Latn) is used.
        tokenizer.src_lang = source_lang
        tokenizer.tgt_lang = target_lang
        return tokenizer(
            batch[get_lang_abbrev(source_lang)],
            text_target=batch[get_lang_abbrev(target_lang)],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
    return tokenize_helper

def _qs_tokenize_fst_fn(
        tokenizer: NllbTokenizer,
        source_lang: Literal['spa_Latn', 'quy_Latn'],
        max_length: int = 512
):

    def tokenize_helper(batch: Any):
        qu_batch = batch['qu']
        es_batch = batch['es']

        tokenizer.src_lang = QUECHUA_LANG_ID
        qu_batch = [encode_text(t, tokenizer) for t in qu_batch]
        qu_encoded = tokenizer(
            qu_batch,
            padding=False,
            truncation=True,
            max_length=max_length,
            is_split_into_words=True,
            add_special_tokens=True,
        )

        tokenizer.src_lang = SPANISH_LANG_ID
        es_encoded = tokenizer(
            es_batch,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )

        if source_lang == QUECHUA_LANG_ID:
            return {
                'input_ids': qu_encoded['input_ids'],
                'attention_mask': qu_encoded['attention_mask'],
                'labels': es_encoded['input_ids'],
            }
        else:
            return {
                'input_ids': es_encoded['input_ids'],
                'attention_mask': es_encoded['attention_mask'],
                'labels': qu_encoded['input_ids'],
            }

    return tokenize_helper

# Dataset structure

# quechua_only_dataset_id = 'Llamacha/monolingual-quechua-iic'
# print(q_dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 175408
#     })
# })

# quechua_spanish_dataset_id = 'somosnlp-hackathon-2022/spanish-to-quechua'
# print(qs_dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['es', 'qu'],
#         num_rows: 102747
#     })
#     validation: Dataset({
#         features: ['es', 'qu'],
#         num_rows: 12844
#     })
#     test: Dataset({
#         features: ['es', 'qu'],
#         num_rows: 12843
#     })
# })


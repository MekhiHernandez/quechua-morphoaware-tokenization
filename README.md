# Morphologically Aware Tokenization Finetuning of NLLB for Spanish-Quechua Translation

Finetuning Meta's [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) 
translation model to improve Spanish ↔ Quechua translation by replacing NLLB's default subword 
tokenization for Quechua with morpheme tokenization from a finite state transducer (FST).

> **Status:** Work in progress. The training and evaluation is 
> functional, but results are still being tuned and the code is 
> under active development. Expect rough edges.

## Motivation

Quechua is an agglutinative language where words are built by concatenating many 
morphemes which makes standard subword tokenizers a poor fit. 
We hypothesize that grounding tokenization in Quechua's morphological structure will make
it easier for the model to encode meaning into its vocabulary and elicit better performance.

## Approach

1. **Morpheme segmentation.** Quechua words are run through a modified version of Annette Rios's 
   `analyzeCuzco` FST, which splits them into their constituent morphemes. Words 
   the FST doesn't recognize fall back to NLLB's original tokenizer.
2. **Vocabulary extension.** All unique morphemes produced by the FST across the 
   dataset are added as new tokens to NLLB's tokenizer. Each new token's 
   embedding is initialized as the average of the subword embeddings NLLB would 
   have originally produced for that morpheme.
3. **Targeted finetuning.** The encoder is frozen entirely, and the embeddings 
   of all pre-existing (non-FST) tokens are frozen via a gradient hook. Only the 
   new morpheme embeddings and the decoder are updated, concentrating learning 
   on the new vocabulary without disturbing NLLB's existing knowledge.
4. **Evaluation.** BLEU, chrF, and chrF++ on the 
   [somosnlp-hackathon-2022/spanish-to-quechua](https://huggingface.co/datasets/somosnlp-hackathon-2022/spanish-to-quechua) 
   dataset.

## Repository structure

- `initial_load_model.py` - downloads and caches the base NLLB model
- `extend_model_vocabulary.py` - extracts FST morphemes and extends the tokenizer + embedding matrix
- `train_fst_model.py` - finetunes the extended model
- `evaluate_base_model.py` - evaluates the base model on the test set
- `sample_translations.py` - prints sample translations
- `common/` - shared utilities (tokenization, FST wrapper, training loop, evaluation)

## Running it

```bash
pip install -r requirements.txt
python initial_load_model.py        # download NLLB-200 distilled 600M
python extend_model_vocabulary.py   # add FST morphemes to vocabulary
python train_fst_model.py           # finetune
python evaluate_base_model.py       # evaluate on test set
```

Training uses gradient accumulation and gradient checkpointing and fits in 
roughly 8 GB of VRAM at the default settings.

To reproduce the training environment on an AWS EC2 GPU instance (Deep Learning AMI), 
see [docs/aws-training-setup.md](docs/aws-training-setup.md).

## Morphological analyzer

Morpheme segmentation is performed by a modified version of the `analyzeCuzco` FST from Annette Rios's 
Quechua language toolkit, which targets the Southern Quechua (Cuzco) variety.

If you use this project, please cite the underlying analyzer:

- Rios Gonzales, A. (2016). A basic language technology toolkit for Quechua. 
  *Procesamiento del Lenguaje Natural*, 56, 91–94.
- Rios Gonzales, A., & Castro Mamani, R. A. (2014). Morphological Disambiguation
  and Text Normalization for Southern Quechua Varieties. In *Proceedings of the
  First Workshop on Applying NLP Tools to Similar Languages, Varieties and
  Dialects (VarDial)*, pages 39–47, Dublin, Ireland.

## Related work

This project is in the same research vein as prior work on morphologically 
informed segmentation for low resource NMT (e.g., Ortega et al.'s BPE-Guided 
segmentation for Quechua, 2021). The contribution here is using the FST output directly 
as new vocabulary entries in an NLLB-scale pretrained model, rather than as a 
segmentation target for learned subword methods.

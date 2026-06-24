"""
Standalone prototype + validation for SURFACE-preserving FST segmentation.

Segments real dataset Quechua by reading the FST's INPUT (surface) tape and cutting at
output morpheme symbols (the "symbol-first" rule confirmed by inspect_fst_alignment.py),
then reports:
  - FST coverage: fraction of word tokens the analyzer recognizes (rest fall back to NLLB)
  - round-trip losslessness: does strip-markers+concat reproduce the EXACT original word?
  - normalization fixed: how often the old (output-tape) root differs from the surface,
    i.e. cases this approach repairs
  - segment-count distribution + examples

This does NOT modify the pipeline; it only measures whether surface segmentation is
lossless and high-coverage enough to justify re-extracting vocab + retraining.

Run from repo root (needs pynini + datasets):
    PYTHONPATH=. python scripts/prototype_surface_segmentation.py [N_SENTENCES]
"""
import sys
from collections import Counter

import pynini
from datasets import load_dataset

from common.process_word_windows import (
    fst, input_symbols, output_symbols, EPSILON_IDX, run_fst, TOKEN_RE,
    SMALLEST_CHUNK_FOR_FST,
)

DATASET_ID = 'somosnlp-hackathon-2022/spanish-to-quechua'


def segment_surface(word: str):
    """Return list of (marker, surface_substring) using the input tape, or None if the
    FST doesn't recognize the word. marker is '=' (root/word-initial) or '+' (suffix)."""
    try:
        acceptor = pynini.accep(' '.join(word), token_type=input_symbols)
    except pynini.FstStringCompilationError:
        return None
    lattice = pynini.shortestpath(acceptor @ fst)
    if lattice.num_states() == 0:
        return None

    segs: list[list[str]] = []   # [marker, surface]
    leading = ''
    state = lattice.start()
    while state != -1:
        arc = next(iter(lattice.arcs(state)), None)
        if arc is None:
            break
        ich = input_symbols.find(arc.ilabel) if arc.ilabel != EPSILON_IDX else ''
        if arc.olabel != EPSILON_IDX:
            osym = output_symbols.find(arc.olabel)
            marker = '+' if osym.startswith('+') else '='
            segs.append([marker, leading + ich])
            leading = ''
        else:
            if segs:
                segs[-1][1] += ich
            else:
                leading += ich
        state = arc.nextstate
    if leading and segs:           # trailing chars with no further morpheme symbol
        segs[-1][1] += leading
    return [(m, s) for m, s in segs]


def core_of(tok: str) -> str:
    lead = len(tok) - len(tok.lstrip("'"))
    trail = len(tok) - len(tok.rstrip("'"))
    return tok[lead:len(tok) - trail] if trail else tok[lead:]


def main(n_sentences: int) -> None:
    ds = load_dataset(DATASET_ID)['validation']
    n_sentences = min(n_sentences, len(ds))
    print(f'sampling {n_sentences} Quechua sentences from {DATASET_ID} [validation]\n')

    total_words = recognized = lossless = norm_fixed = 0
    seg_counts: Counter = Counter()
    fix_examples: list[str] = []
    loss_examples: list[str] = []

    for row in ds.select(range(n_sentences)):
        for m in TOKEN_RE.finditer(row['qu'].replace('’', "'")):
            tok = m.group(0)
            if not any(c.isalpha() for c in tok):
                continue
            word = core_of(tok)
            if len(word) <= SMALLEST_CHUNK_FOR_FST:
                continue
            total_words += 1

            segs = segment_surface(word)
            if segs is None:
                continue
            recognized += 1
            seg_counts[len(segs)] += 1

            rebuilt = ''.join(s for _, s in segs)         # strip markers + concat
            if rebuilt == word:
                lossless += 1
            elif len(loss_examples) < 15:
                loss_examples.append(f'{word!r} -> {segs} -> {rebuilt!r}')

            old = run_fst(word)                            # old (output-tape) result
            old_joined = ''.join(old).replace('=', '').replace('+', '') if old else ''
            if old_joined and old_joined != word:
                norm_fixed += 1
                if len(fix_examples) < 15:
                    surf = ''.join(f'{mk}{s}' for mk, s in segs)
                    fix_examples.append(f'{word!r}: old={old_joined!r}  new(surface)={surf!r}')

    print('==================== RESULTS ====================')
    print(f'word tokens examined        : {total_words}')
    print(f'FST-recognized (else NLLB)  : {recognized}  ({recognized/max(total_words,1):.1%})')
    print(f'round-trip LOSSLESS          : {lossless}/{recognized}  ({lossless/max(recognized,1):.2%})')
    print(f'normalization FIXED vs old   : {norm_fixed}/{recognized}  ({norm_fixed/max(recognized,1):.1%})')
    print(f'segments/word distribution   : {dict(sorted(seg_counts.items()))}')

    print('\n--- examples where surface segmentation REPAIRS the old normalized root ---')
    for e in fix_examples:
        print('   ', e)

    if loss_examples:
        print('\n--- !! round-trip LOSSY cases (should be empty) ---')
        for e in loss_examples:
            print('   ', e)
    else:
        print('\n(no lossy cases: every recognized word reconstructs exactly)')


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    main(n)

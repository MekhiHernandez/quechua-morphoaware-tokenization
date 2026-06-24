"""
Read-only diagnostic: dump the arc-by-arc (input-char, output-symbol) alignment that
analyzeCuzco produces for a few words.

Why: we want a SURFACE-preserving segmenter. run_fst currently reads the OUTPUT tape
(normalized/glossed analysis). We want to instead read the INPUT tape (the original
surface characters) and cut a morpheme boundary wherever the output emits a morpheme
symbol. To place the cut correctly we must know whether the FST emits each morpheme
symbol BEFORE, AFTER, or interspersed with consuming that morpheme's characters.

This also confirms two things:
  (1) the input tape really is the exact surface input (no normalization), and
  (2) the output tape still delimits morphemes (so a boundary signal exists).

Run where pynini + analyzeCuzco.windows.fst are available (e.g. the training box):
    python scripts/inspect_fst_alignment.py
"""
import pynini
from common.process_word_windows import fst, input_symbols, output_symbols, EPSILON_IDX

# Mix of categories. Swap in real dataset words to be sure they're in-distribution.
SAMPLES = [
    'wasikuna',    # house + plural   -> expect surface w,a,s,i | k,u,n,a with =root/+suffix on output
    'wasipi',      # house + locative
    'warmakuna',   # child + plural
    'rimani',      # speak-1sg (verb)
    'mikhuni',     # eat-1sg (verb)
    'ñuqa',        # 1sg pronoun (output may be a gloss tag like PrnPers+1.Sg)
    'qusqu',       # Cusco (3-vowel)
    'llaqtakunapi',# town + plural + locative (longer, multi-suffix)
]


def dump_alignment(word: str) -> None:
    try:
        acceptor = pynini.accep(' '.join(word), token_type=input_symbols)
    except pynini.FstStringCompilationError:
        print('  !! could not build acceptor (a char is not in the FST input alphabet)')
        return
    lattice = pynini.shortestpath(acceptor @ fst)
    if lattice.num_states() == 0:
        print('  (not recognized by FST)')
        return

    state = lattice.start()
    surface_chars: list[str] = []
    out_syms: list[str] = []
    print(f'  {"in (surface)":<16}{"out (analysis)"}')
    print(f'  {"-" * 14}  {"-" * 22}')
    while state != -1:
        arc = next(iter(lattice.arcs(state)), None)
        if arc is None:
            break
        in_ch = input_symbols.find(arc.ilabel) if arc.ilabel != EPSILON_IDX else 'ε'
        out_sym = output_symbols.find(arc.olabel) if arc.olabel != EPSILON_IDX else 'ε'
        if arc.ilabel != EPSILON_IDX:
            surface_chars.append(in_ch)
        if arc.olabel != EPSILON_IDX:
            out_syms.append(out_sym)
        print(f'  {in_ch!r:<16}{out_sym!r}')
        state = arc.nextstate

    print(f'  -> surface reconstructed from INPUT tape : {"".join(surface_chars)!r}')
    print(f'  -> output symbols (current run_fst result): {out_syms}')


if __name__ == '__main__':
    for w in SAMPLES:
        print(f'\nWORD: {w!r}')
        dump_alignment(w)

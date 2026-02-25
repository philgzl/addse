#!/bin/bash
uv run addse ldopt \
    data/external/EARS/ \
    data/external/LibriSpeech/ \
    data/external/VCTK/ \
    data/external/DNS/datasets_fullband/clean_fullband/ \
    data/external/MLS_URGENT_2025_track1/ \
    --regexes "^.*/p0[0-9][0-9]/.*\.wav$" \
    --regexes "^.*/train-clean-(100|360)/.*\.flac$" \
    --regexes "^.*\.flac$" \
    --regexes "^.*\.wav$" \
    --regexes "^.*\.flac$" \
    --labels ears \
    --labels libri \
    --labels vctk \
    --labels dns \
    --labels mls \
    data/chunks/bigspeech/ \
    --num-workers 4 \
    --seed 42

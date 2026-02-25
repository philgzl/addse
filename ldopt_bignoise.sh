#!/bin/bash
uv run addse ldopt \
    data/external/WHAM_48kHz/ \
    data/external/DEMAND/ \
    data/external/FSD50K/ \
    data/external/DNS/datasets_fullband/noise_fullband/ \
    data/external/FMA_medium/ \
    --regexes "^.*/file(0[0-9][0-9]|1[0-7][0-9]|18[0-8]).*\.wav$" \
    --regexes "^.*\.wav$" \
    --regexes "^.*/dev_audio/.*\.wav$" \
    --regexes "^.*\.wav$" \
    --regexes "^.*\.mp3$" \
    --labels wham \
    --labels demand \
    --labels fsd50k \
    --labels dns \
    --labels fma \
    --seglens 30.0 \
    --seglens 10.0 \
    --seglens 30.0 \
    --seglens 0.0 \
    --seglens 0.0 \
    data/chunks/bignoise/ \
    --num-workers 4 \
    --seed 42

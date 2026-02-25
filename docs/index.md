# addse

Code for "Absorbing Discrete Diffusion for Speech Enhancement".

---

[**Documentation**](https://addse.philgzl.com/)

[**Repository**](https://github.com/philgzl/addse)

[**Audio examples**](https://philgzl.com/addse-demo)

---

## Installation

* Install [uv](https://docs.astral.sh/uv/)

* Clone the repository and install dependencies:

    ```bash
    git clone git@github.com:philgzl/addse.git && cd addse && uv sync
    ```

## Data preparation

### Training data

You must first download the following datasets:

- **Speech**: EARS, LibriSpeech, VCTK, DNS, MLS_URGENT_2025_track1
- **Noise**: WHAM_48kHz, DEMAND, FSD50K, DNS, FMA_medium

Place each dataset under `data/external/`. Then run the following scripts:

```bash
./ldopt_bigspeech.sh
./ldopt_bignoise.sh
```

This converts the data to an optimized format for `litdata` and writes it in `data/chunks/`.

Alternatively, update the two shell scripts to use your own speech and noise data.

### Validation data

Validation data is directly streamed from HuggingFace. No need to prepare anything.

Alternatively, update the configuration files in `configs/` to use your own `litdata`-optimized validation data.

### Evaluation data

You must download the Clarity speech dataset to `data/external/Clarity/`. Then run:

```bash
uv run addse ldopt data/external/Clarity/ data/chunks/clarity/ --num-workers 4
```

The remaining evaluation data is directly streamed from HuggingFace.

Alternatively, update the configuration files in `configs/` to use your own `litdata`-optimized evaluation data.

## Training

To train a model:

```bash
uv run addse train configs/<model_name>.yaml
```

Checkpoints and metrics are written to `logs/<model_name>/`.

You can also use the `--wandb` option to log metrics to W&B, and the `--log_model` option to additionally upload checkpoints to W&B, after configuring a `.env` with your credentials.

## Evaluation

To evaluate a trained model:

```bash
uv run addse eval configs/<model_name>.yaml logs/<model_name>/checkpoints/last.ckpt --num-consumers 4
```

The results are written in `eval.db` by default.

## Trained checkpoints

Will be released soon.

## Citation

Paper coming soon.

# WGAN-GP for Multifractal Texture Generation

WGAN-GP with a SAGAN-style self-attention block, trained on 2-channel multifractal
texture data (`.mat` files) and evaluated with a downstream MATLAB/Octave
wavelet-based multifractal analysis pipeline.

This repo is a refactor of an exploratory Jupyter notebook into runnable,
testable modules. All logic lives in `src/`; notebooks are for interactive
exploration only.

## Repo structure

```
configs/
  base.yaml               # shared hyperparameters
  experiments/*.yaml       # per-run overrides (one file per experiment, instead
                           # of copy-pasted notebook cells)
src/
  data.py                  # MatDataset, train/val split
  models/                  # SelfAttention, Generator, Critic, weight init
  losses.py                # WGAN-GP gradient penalty
  training.py               # training loop, checkpointing, resume
  octave_io.py              # export generated samples to Octave .mat files
  analysis/                 # shift/rotation correlation "did it memorize?" check
scripts/
  train.py                  # CLI: python scripts/train.py --config ...
  generate_samples.py        # export samples from a trained checkpoint
  run_overfit_check.py        # run the memorization check end-to-end
notebooks/
  exploration.ipynb          # thin notebook, imports from src/
tests/
  test_shapes.py             # model/loss shape sanity checks
```

## Setup

Requires PyTorch built for CUDA 12.8 if running on Blackwell-generation GPUs
(e.g. RTX 5090):

```bash
conda env create -f environment.yml
conda activate gans310
```

or with plain pip (adjust the torch index URL as needed for your GPU):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Data

Point `data_root` in a config at a folder of `.mat` files, each containing a
`datatmp` variable holding a complex-valued `(H, W, C)` array (only the real
part is used). See `configs/base.yaml` for the default key/path.

## Training

1. Copy `configs/experiments/att-1.yaml` (or edit it) to set `output_dir`,
   `batch_size`, `num_epochs`, etc. Only include keys that differ from
   `configs/base.yaml`.
2. Run:

```bash
python scripts/train.py --config configs/experiments/att-1.yaml
```

This creates `<output_dir>/{samples,checkpoints,matlab_data}/`, trains with
resume-from-latest-checkpoint support (`resume_training: true`), saves PNG
previews and checkpoints every `save_epoch` epochs, and exports generated
textures to Octave-compatible `.mat` files at 128/512/1024 resolution for the
MATLAB multifractal analysis pipeline.

To launch a sweep of experiments, add more files under `configs/experiments/`
and loop over them from a shell script — no need to edit Python.

## Generating samples from a trained checkpoint

```bash
python scripts/generate_samples.py \
    --output-dir /path/to/experiment \
    --epoch 25 --resolution 512 --sample-size 10
```

## Checking for memorization (overfitting)

Compares a generated sample against every validation image across a grid of
shifts and 4 rotations, looking for suspiciously high Pearson correlation
with any single training image:

```bash
python scripts/run_overfit_check.py \
    --checkpoint /path/to/experiment/checkpoints/generator_epoch_25.pth \
    --data-root /path/to/dataV6norm/
```

## Tests

```bash
pytest tests/
```

## Notes

- Generator input noise uses spatial dimensions `(noise_height + 14, noise_width + 14)`;
  the first conv (`kernel_size=15`, no padding) removes exactly that 14px pad, then
  three x2 upsampling stages give a final resolution of `noise_height * 8`.
- `.mat`, `checkpoints/`, `samples/`, and `matlab_data/` are gitignored — these are
  large generated artifacts, not source.

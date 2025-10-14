# nanochat

[Original README](README.md)

## documents

- [The tutorial](https://github.com/karpathy/nanochat/discussions/1)
- [huggingface nanochat students](https://huggingface.co/nanochat-students)

## setup

create `env.sh`

```bash
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=~/.cache/nanochat
export WANDB_RUN=dummy  # adjust to log to wandb
```


```bash
conda update conda -n base -y
rustup update
conda create -n nc python=3.13 -y
conda activate nc
pip install torch torchvision
pip install -e .
pip install "maturin<2"
maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## commands

```bash
# download
python -m nanochat.dataset -n 240

# reset the markdown report output
python -m nanochat.report reset

# train tokenizer
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval
```

## continue at pretraining


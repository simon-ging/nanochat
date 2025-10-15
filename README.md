# nanochat

[Original README](README.md)

## changes

- conda instead of uv

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
source env.sh
conda update conda -n base -y
rustup update
conda create -n nc python=3.13 -y
conda activate nc
pip install torch torchvision
pip install -e .
pip install "maturin<2" wandb "black[jupyter]"
maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## commands

```bash
# download training data (-n 240 for less data)
python -m nanochat.dataset

# download eval data
cd $NANOCHAT_BASE_DIR
curl -L -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
unzip -q eval_bundle.zip
rm eval_bundle.zip
cd -

# reset the markdown report output
python -m nanochat.report reset

# train tokenizer
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# pretrain
# author recommended bs 32 for h100
export WANDB_RUN=run1
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --device_batch_size 64 --depth=30

# eval
torchrun --standalone --nproc_per_node=4 -m scripts.base_loss
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval

# mid train and eval
torchrun --standalone --nproc_per_node=4 -m scripts.mid_train
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i mid

# sft train and eval
torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i sft

# rl train and eval
torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i rl -a GSM8K
```


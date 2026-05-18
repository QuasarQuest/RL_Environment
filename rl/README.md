# ATB — RL Training Pipeline

## Structure

```
rl/
├── src/
│   ├── env/
│   │   ├── atb_env.py      # Gymnasium wrapper around Rust sim
│   │   └── wrappers.py     # Reward scaling, clipping
│   ├── model/
│   │   ├── policy.py       # MLP architecture + policy kwargs
│   │   └── export.py       # ONNX export for Rust inference
│   ├── training/
│   │   ├── config.py       # All hyperparameters
│   │   ├── train.py        # PPO entry point
│   │   ├── tune.py         # Optuna hyperparameter search
│   │   └── callbacks.py    # Checkpointing + HDF5/TB stat logging
│   └── utils/
│       └── logger.py       # Read HDF5 stats into Pandas
├── runs/                   # gitignored
│   ├── models/             # .zip checkpoints + .onnx exports
│   ├── stats/              # HDF5 episode stats + Optuna DB
│   └── tensorboard/        # TensorBoard event files
└── pyproject.toml
```

## Setup

```bash
cd rl
python -m venv .venv
source .venv/bin/activate
pip install maturin[patchelf]
pip install stable-baselines3[extra] optuna optuna-integration h5py onnx onnxruntime rich typer pandas
cd ..
maturin develop --release --features python
```

## Train

```bash
python -m src.training.train
python -m src.training.train --total-timesteps 5000000 --run-name exp1
python -m src.training.train --resume runs/models/run_best.zip
```

## Monitor

```bash
tensorboard --logdir runs/tensorboard
```

## Tune

```bash
python -m src.training.tune --n-trials 50 --n-timesteps 500000
```

## Export to ONNX (for Rust)

```bash
python -m src.model.export --model runs/models/run_best.zip --out runs/models/policy.onnx
```

## Analyse stats

```python
from src.utils.logger import read_stats, print_summary
print_summary("runs/stats/run_123.h5")
df = read_stats("runs/stats/run_123.h5")
df["episode_reward"].rolling(100).mean().plot()
```
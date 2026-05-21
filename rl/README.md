# ATB RL

Reinforcement learning pipeline for Algorithm Test Bed. PPO via Stable-Baselines3, Hydra config, HDF5 stats.

## Setup

```bash
cd rl/
pip install -e .
```

Run all commands from `rl/`.

## Train

```bash
atb-train                                              # defaults: ppo, stage1, 2M steps
atb-train train.total_timesteps=5000000                # override a value
atb-train ppo=aggressive                               # swap PPO profile
atb-train ppo=aggressive env=stage2                    # aggressive + stage 2
atb-train +train.resume=runs/models/run_s1_XYZ_final  # resume
atb-train --multirun ppo.learning_rate=1e-4,3e-4       # grid search
```

## Monitor (second terminal while training)

```bash
atb-monitor runs/stats/run_s1_<timestamp>.h5
```

## Analyse

```bash
atb-stats summary runs/stats/run_s1_<timestamp>.h5   # single run numbers
atb-stats compare runs/stats/                        # compare all runs
atb-plot runs/stats/run_s1_<timestamp>.h5            # reward/length/score/win plots
atb-plot runs/stats/ --compare                       # overlay all runs
tensorboard --logdir runs/tensorboard
```

## Tune (after reward signal is stable)

```bash
atb-tune --stage 1 --n-trials 50 --n-timesteps 500000
```

## Export

```bash
atb-export --model runs/models/run_s1_XYZ_final --out runs/exports/model.onnx
```

## Output layout

```
runs/
  models/       checkpoints + final model
  stats/        HDF5 episode data
  tensorboard/  TensorBoard logs
  outputs/      Hydra logs + resolved configs per run
```

## Config

Configs live in `configs/`. Override anything on the CLI with `key=value` or swap a whole group with `ppo=aggressive`. To see the full resolved config for a run check `runs/outputs/`.

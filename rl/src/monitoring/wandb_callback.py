"""Optional Weights & Biases integration.

Self-contained — train.py stays clean. Add two lines when you want it:

    from monitoring.wandb_callback import WandbCallback, init_wandb
    init_wandb(cfg, run_tag)
    callbacks = [..., WandbCallback()]

Setup
-----
    pip install wandb
    wandb login

What gets logged
----------------
- train/episode_reward, episode_length, score  (per episode)
- train/entropy_loss, policy_loss, value_loss, approx_kl (SB3 built-ins)
- train/ent_coef                                          (scheduled value)
- perf/steps_per_sec
- Full ATBConfig as run config (searchable in the wandb UI)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from omegaconf import DictConfig


def init_wandb(
    cfg: "DictConfig",
    run_tag: str,
    project: str = "atb",
    entity: Optional[str] = None,
) -> None:
    """Initialise a wandb run. Call once before model construction.

    Parameters
    ----------
    cfg      : the raw DictConfig from @hydra.main (not the typed ATBConfig)
    run_tag  : timestamped run identifier from train.py
    project  : wandb project name
    entity   : wandb team/user — None uses your default
    """
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Run: pip install wandb") from exc

    from omegaconf import OmegaConf

    wandb.init(
        project=project,
        entity=entity,
        name=run_tag,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=False,
        reinit=True,
    )


class WandbCallback(BaseCallback):
    """Mirror episode stats and SB3 train metrics to Weights & Biases.

    Sits alongside existing callbacks — HDF5 and TensorBoard keep working.

    Parameters
    ----------
    save_model      : upload the final model .zip as a wandb artifact
    model_save_path : path where train.py saves the final model
    """

    def __init__(
        self,
        save_model: bool = False,
        model_save_path: Optional[Path] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._save_model = save_model
        self._model_save_path = model_save_path
        self._start_time = time.time()

    def _on_step(self) -> bool:
        try:
            import wandb
        except ImportError:
            return True

        log: dict = {}

        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            ep = info["episode"]
            log["train/episode_reward"] = float(ep["r"])
            log["train/episode_length"] = int(ep["l"])
            log["train/score"] = float(info.get("score", 0.0))

        for key in (
            "train/entropy_loss", "train/policy_loss", "train/value_loss",
            "train/approx_kl", "train/clip_fraction", "train/explained_variance",
        ):
            val = self.logger.name_to_value.get(key)
            if val is not None:
                log[key] = float(val)

        if hasattr(self.model, "ent_coef"):
            log["train/ent_coef"] = float(self.model.ent_coef)

        elapsed = max(time.time() - self._start_time, 1.0)
        log["perf/steps_per_sec"] = self.num_timesteps / elapsed

        if log:
            wandb.log(log, step=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        try:
            import wandb
        except ImportError:
            return

        if self._save_model and self._model_save_path is not None:
            model_zip = Path(str(self._model_save_path) + ".zip")
            if model_zip.exists():
                artifact = wandb.Artifact(name=f"model-{wandb.run.name}", type="model")  # type: ignore[union-attr]
                artifact.add_file(str(model_zip))
                wandb.log_artifact(artifact)
                if self.verbose:
                    print(f"  ✓ wandb artifact: {model_zip.name}")

        wandb.finish()

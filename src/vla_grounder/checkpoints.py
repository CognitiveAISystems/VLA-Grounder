from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback


class SaveEveryCheckpointCallback(TrainerCallback):
    def __init__(self, save_every: int):
        if save_every <= 0:
            raise ValueError("save_every must be positive")
        self.save_every = save_every
        self._trainer = None
        self._last_saved_step = 0

    def set_trainer(self, trainer) -> None:
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        should_save = (
            self._trainer is not None
            and step > 0
            and step % self.save_every == 0
            and step != self._last_saved_step
        )
        if should_save:
            checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            self._trainer.save_model(checkpoint)
            self._last_saved_step = step
        return control

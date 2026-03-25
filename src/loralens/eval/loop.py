# src/loralens/eval/loop.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Eval:
    """
    Minimal stub evaluation command so the CLI can import Eval and parse
    Main.command: Union[Train, Eval].

    This just prints a message for now; you'll replace this with a real
    evaluation loop later.
    """

    model_name: str = "sshleifer/tiny-gpt2"
    checkpoint_path: Optional[str] = None

    def execute(self) -> None:
        raise NotImplementedError(
            "Eval loop is not implemented yet. "
            "For now, use --command.train.* arguments to run training."
        )
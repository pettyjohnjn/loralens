import logging
from dataclasses import dataclass
from typing import Union, Literal, Optional

from simple_parsing import ArgumentParser, ConflictResolution
from loralens.training.loop import Train
from loralens.eval.loop import Eval

@dataclass
class Main:
    command: Union[Train, Eval]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def execute(self):
        logging.basicConfig(level=self.log_level, format="[%(levelname)s] %(message)s")
        self.command.execute()

def main(args: Optional[list[str]] = None):
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(Main, dest="prog")
    args = parser.parse_args(args=args)
    prog: Main = args.prog
    prog.execute()


if __name__ == "__main__":
    main()
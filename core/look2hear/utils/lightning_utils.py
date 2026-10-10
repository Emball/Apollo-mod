from rich.console import Console
from pytorch_lightning.utilities import rank_zero_only

_console = Console(highlight=False, markup=False)


@rank_zero_only
def print_only(message: str):
    _console.print(message)

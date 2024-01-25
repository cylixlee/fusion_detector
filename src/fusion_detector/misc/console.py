import colorama
from colorama import Fore, Back

__all__ = ["message", "success", "error"]

colorama.init(autoreset=True)


def message(text: str) -> None:
    print(Fore.BLACK + Back.BLUE + " MESSAGE ", text)


def success(text: str) -> None:
    print(Fore.BLACK + Back.GREEN + " SUCCESS ", text)


def error(text: str) -> None:
    print(Fore.BLACK + Back.RED + " ERROR!! ", text)

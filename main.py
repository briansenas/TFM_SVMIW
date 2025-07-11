from __future__ import annotations

import argparse
import importlib
import os
import pkgutil

from scripts.lib.utils import load_yaml_defaults


def load_subparsers(subparsers, scripts_package: str = "scripts"):
    """
    Dynamically load and register subparsers from modules in the 'scripts' package.
    Each module must define a `register_subparser(subparsers)` function.
    """
    package_path = os.path.join(os.path.dirname(__file__), scripts_package)
    subparsers_map = {}
    for _, name, _ in pkgutil.iter_modules([package_path]):
        module_name = f"{scripts_package}.{name}"
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            print(f"Failed to import '{module_name}': {e}")
            continue
        if name != "lib":
            register_func = getattr(module, "register_subparser", None)
            if callable(register_func):
                command_name = getattr(module, "COMMAND_NAME")
                try:
                    parser = register_func(subparsers)
                    subparsers_map[command_name] = parser
                except (TypeError, ValueError, argparse.ArgumentError) as e:
                    print(f"Failed to register subparser in '{module_name}': {e}")
            else:
                print(
                    f"Warning: '{module_name}' does not define a callable 'register_subparser' function.",
                )

    return subparsers_map


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers_map = load_subparsers(subparsers)
    args, _ = parser.parse_known_args()
    # Load defaults into the subparser if config is given
    if args.config and args.command:
        subparser = subparsers_map[args.command]
        load_yaml_defaults(subparser, args.config)
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)


if __name__ == "__main__":
    main()

"""Command-line entry point; importing this module never opens a window."""

if __package__:
    from .cli import main
else:
    # Also support the editor's "run Python file" action without opening a GUI.
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from barnacle_challenge.cli import main


if __name__ == "__main__":
    main()

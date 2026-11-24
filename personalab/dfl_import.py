"""Validate DeepFaceLab JPEG datasets and write a JSON summary."""

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import TextIO

from .dfl import DatasetError, validate_dataset


def _write_json(report: dict, stream: TextIO) -> None:
    json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")


def _write_file(report: dict, output: Path) -> None:
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=".dfl-validation-", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            _write_json(report, stream)
            stream.close()
            os.link(temporary, output)
        finally:
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JPEG file, directory or ZIP")
    parser.add_argument("--output", type=Path, help="new JSON report path (default: stdout)")
    args = parser.parse_args(argv)

    try:
        if (
            args.output is not None and args.input.is_dir()
            and args.output.resolve().is_relative_to(args.input.resolve())
        ):
            parser.exit(2, f"{parser.prog}: the report must be outside the dataset directory\n")
        try:
            report = validate_dataset(args.input)
            payload = report.to_dict()
            status = int(report.rejected_samples != 0)
        except DatasetError as error:
            payload = {"dataset_error": {"code": error.code, "message": str(error)}}
            status = 2
        if args.output is None:
            _write_json(payload, sys.stdout)
        else:
            _write_file(payload, args.output)
    except (OSError, RuntimeError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

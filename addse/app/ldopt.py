import functools
import importlib.resources
import logging
import random
from collections.abc import Iterator
from typing import Annotated

import litdata as ld
import soundfile as sf
import typer
import yaml

from ..utils import scan_files, segment_audio_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = typer.Typer()


def ldopt_segment_audio_file(
    file: tuple[str, str, float, str],
    format: str,
    subtype: str,
) -> Iterator[tuple[bytes, str]]:
    """Unpack a tuple of file, input directory, segment length, label, and call `segment_audio_file`."""
    base, path, seglen, label = file
    try:
        for bytes, name in segment_audio_file(
            path=path, format=format, subtype=subtype, seglen=None if seglen == 0.0 else seglen, base=base
        ):
            yield bytes, name if label == "" else f"{label}:{name}"
    except sf.LibsndfileError as e:
        logger.warning(f"Could not process {path}: {e}")


@app.command()
def ldopt(
    input_dirs: Annotated[list[str], typer.Argument(help="Directory with audio files to optimize.")],
    output_dir: Annotated[str, typer.Argument(help="Directory to save optimized files.")],
    regexes: Annotated[list[str], typer.Option(help=r"Regex to filter files.")] = [r"^.*\.wav$"],
    num_workers: Annotated[int, typer.Option(help="Number of workers.")] = 4,
    chunk_bytes: Annotated[str, typer.Option(help="Chunk size in bytes.")] = "64MB",
    seglens: Annotated[list[float], typer.Option(help="Segment length in seconds.")] = [0.0],
    format: Annotated[str, typer.Option(help="Audio format to convert to.")] = "ogg",
    subtype: Annotated[str, typer.Option(help="Audio subtype to convert to.")] = "opus",
    labels: Annotated[list[str], typer.Option(help="Label replacing the input directory in the stored path.")] = [""],
    seed: Annotated[int | None, typer.Option(help="Random seed for shuffling files.")] = None,
    corpus: Annotated[
        tuple[str, str] | None,
        typer.Option(help="Corpus name and split. Overrides --regex and --seglen. See corpora.yaml."),
    ] = None,
) -> None:
    """Optimize audio files for LitData."""
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    if corpus is not None:
        corpus_name, corpus_split = corpus
        with importlib.resources.open_text("addse", "corpora.yaml") as f:
            corpora = yaml.safe_load(f)
        regexes = corpora[corpus_name][corpus_split]
        seglens = corpora[corpus_name].get("seglen", None)
        labels = corpora[corpus_name].get("label", None)

    regexes = [regexes] * len(input_dirs) if isinstance(regexes, str) else regexes
    regexes = [regexes[0]] * len(input_dirs) if len(regexes) == 1 else regexes
    seglens = [seglens] * len(input_dirs) if isinstance(seglens, float) else seglens
    seglens = [seglens[0]] * len(input_dirs) if len(seglens) == 1 else seglens
    labels = [labels] * len(input_dirs) if isinstance(labels, str) else labels
    labels = [labels[0]] * len(input_dirs) if len(labels) == 1 else labels

    filess = []
    for input_dir, regex, seglen, label in zip(input_dirs, regexes, seglens, labels, strict=True):
        logger.info(f"Scanning {input_dir}...")
        files = [(input_dir, file, seglen, label) for file in scan_files(input_dir, regex)]
        assert files, f"Found no files in {input_dir} matching {regex}"
        logger.info(f"Found {len(files)} files matching {regex}")
        filess.append(files)
    flat_files = sorted(file for files in filess for file in files)
    if seed is not None:
        random.seed(seed)
        random.shuffle(flat_files)
    logger.info(f"Found {len(flat_files)} files in total")

    ld.optimize(
        fn=functools.partial(ldopt_segment_audio_file, format=format, subtype=subtype),
        inputs=flat_files,
        output_dir=output_dir,
        num_workers=num_workers,
        chunk_bytes=chunk_bytes,
    )

import functools
import importlib.resources
import logging
import os
from multiprocessing import Pool
from typing import Annotated

import pyarrow as pa
import pyarrow.parquet as pq
import typer
import yaml
from tqdm import tqdm

from ..utils import bytes_str_to_int, scan_files, segment_audio_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = typer.Typer()


def process_files(
    worker_id: int,
    files: list[str],
    input_dir: str,
    output_dir: str,
    prefix: str,
    schema: pa.Schema,
    chunk_bytes: int,
    seglen: float | None,
    format: str,
    subtype: str,
) -> None:
    """Process files and write to Parquet format."""
    outfile_fmt = os.path.join(output_dir, f"{prefix}-{worker_id:03d}-{{:06d}}.parquet")
    totsize = 0
    count = 0
    outfile = outfile_fmt.format(count)
    writer = pq.ParquetWriter(outfile, schema=schema)
    try:
        for file in tqdm(files):
            for bytes, name in segment_audio_file(file, format=format, subtype=subtype, seglen=seglen, base=input_dir):
                record = pa.record_batch([pa.array([bytes]), pa.array([name])], schema=schema)
                size = record.get_total_buffer_size()
                if totsize != 0 and totsize + size > chunk_bytes:
                    writer.close()
                    totsize = 0
                    count += 1
                    outfile = outfile_fmt.format(count)
                    writer = pq.ParquetWriter(outfile, schema=schema)
                writer.write(record)
                totsize += size
    finally:
        writer.close()


def split_file_list_by_size(file_list: list[str], n: int) -> list[list[str]]:
    """Split a list of files evenly by size."""
    totsize = sum(os.path.getsize(f) for f in file_list)
    splits = []
    current_split: list[str] = []
    current_size = 0
    for file in file_list:
        file_size = os.path.getsize(file)
        if current_size + file_size > totsize / n:
            splits.append(current_split)
            current_split = []
            current_size = 0
        current_split.append(file)
        current_size += file_size
    if len(splits) < n:
        splits.append(current_split)
    else:
        splits[-1].extend(current_split)
    assert len(splits) == n
    return splits


@app.command()
def parquetize(
    input_dir: Annotated[str, typer.Argument(help="Directory with audio files to optimize.")],
    output_dir: Annotated[str, typer.Argument(help="Directory to save optimized files.")],
    regex: Annotated[str, typer.Option(help=r"Regex to filter files.")] = r"^.*\.wav$",
    num_workers: Annotated[int, typer.Option(help="Number of workers.")] = 4,
    chunk_bytes: Annotated[str, typer.Option(help="Chunk size in bytes.")] = "64MB",
    seglen: Annotated[float | None, typer.Option(help="Segment length in seconds.")] = None,
    format: Annotated[str, typer.Option(help="Audio format to convert to.")] = "ogg",
    subtype: Annotated[str, typer.Option(help="Audio subtype to convert to.")] = "opus",
    prefix: Annotated[str, typer.Option(help="Prefix for output files.")] = "chunk",
    corpus: Annotated[
        tuple[str, str] | None,
        typer.Option(help="Corpus name and split. Overrides --regex and --seglen. See corpora.yaml."),
    ] = None,
) -> None:
    """Optimize audio files for Hugging Face."""
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    if corpus is not None:
        corpus_name, corpus_split = corpus
        with importlib.resources.open_text("addse", "corpora.yaml") as f:
            corpora = yaml.safe_load(f)
        regex = corpora[corpus_name][corpus_split]
        seglen = corpora[corpus_name].get("seglen", None)

    logger.info("Scanning files...")
    files = sorted(scan_files(input_dir, regex))
    assert files, f"Found no files in {input_dir} matching {regex}"
    logger.info(f"Found {len(files)} files")

    os.makedirs(output_dir, exist_ok=True)

    fn = functools.partial(
        process_files,
        input_dir=input_dir,
        output_dir=output_dir,
        prefix=prefix,
        schema=pa.schema([("audio", pa.binary()), ("name", pa.string())]),
        chunk_bytes=bytes_str_to_int(chunk_bytes),
        seglen=seglen,
        format=format,
        subtype=subtype,
    )

    num_workers = max(1, min(num_workers, len(files)))
    if num_workers > 0:
        file_splits = split_file_list_by_size(files, num_workers)
        with Pool(num_workers) as pool:
            pool.starmap(fn, enumerate(file_splits))
    else:
        fn(0, files)

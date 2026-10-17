#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
# Evaluation splits. "train" is left out on purpose: it is for model training.
SPLITS = ("dev", "test")


def clean_field(value: str | None) -> str:
    return " ".join((value or "").split())


def find_beir_root(path: Path, dataset: str) -> Path:
    candidates = [
        path,
        path / dataset,
        path / dataset.replace("-", ""),
        path / dataset.replace("_", "-"),
    ]
    for candidate in candidates:
        if has_beir_files(candidate):
            return candidate

    for corpus in path.rglob("corpus.jsonl"):
        candidate = corpus.parent
        if has_beir_files(candidate):
            return candidate

    raise FileNotFoundError(f"could not find BEIR files under {path}")


def has_beir_files(path: Path) -> bool:
    return (
        (path / "corpus.jsonl").is_file()
        and (path / "queries.jsonl").is_file()
        and any((path / "qrels" / f"{split}.tsv").is_file() for split in SPLITS)
    )


def download_and_extract(dataset: str, cache_dir: Path, base_url: str) -> Path:
    # "cqadupstack/android" names the "android" subset of the "cqadupstack" archive.
    archive_name, _, subset = dataset.partition("/")
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/{urllib.parse.quote(archive_name)}.zip"
    archive = cache_dir / f"{archive_name}.zip"

    if not archive.is_file():
        print(f"downloading {url}", file=sys.stderr)
        tmp = archive.with_suffix(".zip.tmp")
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(archive)

    root = cache_dir / dataset
    if not has_beir_files(root):
        print(f"extracting {archive}", file=sys.stderr)
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(cache_dir)

    if subset:
        return find_beir_root(cache_dir / archive_name / subset, subset)
    return find_beir_root(cache_dir, dataset)


def load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] == "query-id":
                continue
            if len(row) < 3:
                raise ValueError(f"invalid qrels row in {qrels_path}: {row!r}")
            query_id, corpus_id, score = row[0], row[1], row[2]
            qrels.setdefault(query_id, {})[corpus_id] = int(score)
    return qrels


def corpus_ids(source_dir: Path) -> list[str]:
    with (source_dir / "corpus.jsonl").open(encoding="utf-8") as source:
        return [json.loads(line)["_id"] for line in source]


def write_corpus(source_dir: Path, output_path: Path, keep: set[str] | None = None) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (source_dir / "corpus.jsonl").open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for line in source:
            doc = json.loads(line)
            doc_id = doc["_id"]
            if keep is not None and doc_id not in keep:
                continue
            title = clean_field(doc.get("title"))
            text = clean_field(doc.get("text"))
            body = f"{title} {text}".strip()
            writer.writerow([doc_id, body])
            count += 1
    return count


def write_queries(source_dir: Path, output_path: Path, qrels: dict[str, dict[str, int]]) -> int:
    queries: dict[str, str] = {}
    with (source_dir / "queries.jsonl").open(encoding="utf-8") as source:
        for line in source:
            query = json.loads(line)
            queries[query["_id"]] = clean_field(query.get("text"))

    missing = [query_id for query_id in qrels if query_id not in queries]
    if missing:
        raise ValueError(f"{len(missing)} qrel queries are missing from queries.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for query_id in qrels:
            writer.writerow([query_id, queries[query_id]])
    return len(qrels)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, separators=(",", ":"))
        output.write("\n")


def sample_qrels(
    qrels: dict[str, dict[str, int]], max_queries: int | None, rng: random.Random
) -> dict[str, dict[str, int]]:
    # Keep only queries with at least one relevant document, as BEIR evaluation does.
    qrels = {q: docs for q, docs in qrels.items() if any(g > 0 for g in docs.values())}
    if max_queries is None or len(qrels) <= max_queries:
        return qrels
    chosen = set(rng.sample(sorted(qrels), max_queries))
    return {q: docs for q, docs in qrels.items() if q in chosen}


def convert(
    source_dir: Path,
    output_name: str,
    datasets_dir: Path = Path("datasets"),
    testset_root: Path = Path("testset"),
    max_queries: int | None = None,
    max_docs: int | None = None,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    dataset_path = datasets_dir / f"{output_name}.tsv"
    testset_dir = testset_root / output_name

    split_qrels = {}
    for split in SPLITS:
        path = source_dir / "qrels" / f"{split}.tsv"
        if path.is_file():
            split_qrels[split] = sample_qrels(load_qrels(path), max_queries, rng)

    keep = None
    if max_docs is not None:
        # Keep every judged document for every split, then add random distractors.
        # A smaller corpus is easier, so compare scores only at the same --max-docs.
        judged = {doc for qrels in split_qrels.values() for docs in qrels.values() for doc in docs}
        all_ids = corpus_ids(source_dir)
        present = [doc for doc in all_ids if doc in judged]
        others = [doc for doc in all_ids if doc not in judged]
        room = max(0, max_docs - len(present))
        keep = set(present) | set(rng.sample(others, min(room, len(others))))

    doc_count = write_corpus(source_dir, dataset_path, keep)
    write_json(testset_dir / "collection_map.json", None)
    for split, qrels in split_qrels.items():
        query_count = write_queries(source_dir, testset_dir / f"questions.{split}.tsv", qrels)
        write_json(testset_dir / f"qrels.{split}.json", qrels)
        qrel_count = sum(len(docs) for docs in qrels.values())
        print(
            f"wrote {testset_dir}/questions.{split}.tsv ({query_count} queries, {qrel_count} qrels)"
        )
    print(f"wrote {dataset_path} ({doc_count} docs)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a BEIR dataset into Witchcraft TSV/testset files."
    )
    parser.add_argument("dataset", help="BEIR dataset name, for example trec-covid")
    parser.add_argument(
        "--output-name",
        default=None,
        help="local dataset/testset name; defaults to the dataset name without '-'",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/beir",
        type=Path,
        help="directory used for downloaded BEIR archives and extracted files",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="existing extracted BEIR dataset directory; skips download",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BEIR_BASE_URL,
        help="base URL containing BEIR dataset zip files",
    )
    parser.add_argument("--datasets-dir", default=Path("datasets"), type=Path)
    parser.add_argument("--testset-dir", default=Path("testset"), type=Path)
    parser.add_argument("--max-queries", type=int, help="sample at most this many queries per split")
    parser.add_argument(
        "--max-docs",
        type=int,
        help="keep all judged documents plus random distractors up to this corpus size",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed for query and document samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_name = args.output_name or args.dataset.replace("-", "").replace("/", "-")
    if args.source_dir:
        source_dir = find_beir_root(args.source_dir, args.dataset)
    else:
        source_dir = download_and_extract(args.dataset, args.cache_dir, args.base_url)
    convert(
        source_dir,
        output_name,
        args.datasets_dir,
        args.testset_dir,
        args.max_queries,
        args.max_docs,
        args.seed,
    )


if __name__ == "__main__":
    main()

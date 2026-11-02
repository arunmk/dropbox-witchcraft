#!/usr/bin/env python3
"""Short-document regression set: select the cases, and build the datasets.

Token pruning keeps a fraction of each document's tokens. A short document then
keeps only a few tokens, and those tokens can be generic words that thousands of
other documents also contain. The document then drops out of the top 100
results, although the same search finds the document without pruning.
testset/regression/short-docs/README.md explains the failure mode and the
choices in detail.

Subcommands:
  select  Read saved eval results and write testset/regression/short-docs/cases.jsonl.
          A case is a test query whose relevant documents all have fewer than
          --max-words words, which the reference configuration finds in the top
          100, and which the pruned configuration does not find in the top 100.
  build   Write the evaluable datasets from cases.jsonl, with the original
          corpus of each case as the distractors:
            regress-short-cqa  BEIR CQADupStack (Android)
            regress-short-mix  the public mixed corpus (scripts/fetch_public_corpus.py)

  scripts/short_doc_regression.py select --results bench-results/eval/keep70-check
  scripts/short_doc_regression.py build
  scripts/eval_suite.py run --suite regression --configs default
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "testset" / "regression" / "short-docs" / "cases.jsonl"
DATASETS_DIR = ROOT / "datasets" / "eval"
TESTSET_DIR = ROOT / "testset" / "eval"

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**63 else 2**31 - 1)

# Each source dataset of the eval suite maps to one regression dataset.
SETS = {
    "cqadupstack-android": {
        "name": "regress-short-cqa",
        "corpus": [DATASETS_DIR / "cqadupstack-android.tsv"],
        "doc_source": "BEIR CQADupStack (Android), Stack Exchange, CC BY-SA",
    },
    "public-mix-hard": {
        "name": "regress-short-mix",
        "corpus": [DATASETS_DIR / "public-mix-hard.tsv", ROOT / "datasets" / "public-mix.tsv"],
        "doc_source": None,  # set for each document from its ID prefix
    },
}
MIX_SOURCES = {
    "enron": "Enron email corpus (corbt/enron-emails), public FERC release",
    "wiki": "English Wikipedia (wikimedia/wikipedia 20231101.en), CC BY-SA",
    "code": "GitHub file (mixed licenses)",
}
# Documents with unknown licenses stay out of the tracked case file.
EXCLUDED_SOURCES = {"code"}


def read_tsv(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as f:
        return {row[0]: row[1] for row in csv.reader(f, delimiter="\t") if len(row) >= 2}


def read_per_query(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    return {r[0]: dict(zip(rows[0][1:], map(float, r[1:]))) for r in rows[1:]}


def read_ranks(path: Path) -> dict[str, list[str]]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            query, documents = line.rstrip("\n").split("\t")
            out[query] = [d for d in documents.split(",") if d]
    return out


def select(args) -> int:
    results = Path(args.results)
    cases = []
    for source, spec in SETS.items():
        run_dir = results / "runs" / source / "test"
        if not run_dir.is_dir():
            print(f"{source}: no results in {run_dir}; skipped", file=sys.stderr)
            continue
        docs = read_tsv(DATASETS_DIR / f"{source}.tsv")
        qrels = json.loads((TESTSET_DIR / source / "qrels.test.json").read_text())
        questions = read_tsv(TESTSET_DIR / source / "questions.test.tsv")
        styles = {}
        records = TESTSET_DIR / source / "queries.jsonl"
        if records.is_file():
            styles = {r["query_id"]: r["style"] for r in map(json.loads, records.open(encoding="utf-8"))}
        reference = read_per_query(run_dir / args.reference / "per_query.tsv")
        pruned = read_per_query(run_dir / args.pruned / "per_query.tsv")
        ranks = {c: read_ranks(run_dir / c / "results.tsv")
                 for c in (args.reference, args.pruned, *args.also) if (run_dir / c).is_dir()}
        for query, metrics in sorted(reference.items()):
            relevant = [d for d, grade in qrels.get(query, {}).items() if grade > 0 and d in docs]
            if not relevant or max(len(docs[d].split()) for d in relevant) >= args.max_words:
                continue
            if metrics["recall@100"] < 1.0 or pruned[query]["recall@100"] > 0.0:
                continue
            prefixes = {d.split(":")[0] for d in relevant if ":" in d}
            if prefixes & EXCLUDED_SOURCES:
                continue
            cases.append({
                "case_id": f"{spec['name']}:{query}",
                "set": spec["name"],
                "source_dataset": source,
                "source_query_id": query,
                "query": questions[query],
                "query_style": styles.get(query),
                "relevant": [{
                    "doc_id": d,
                    "words": len(docs[d].split()),
                    "source": spec["doc_source"] or MIX_SOURCES[d.split(":")[0]],
                    "text": docs[d],
                } for d in relevant],
                # The 1-based rank of each relevant document in each configuration; null = not in the top 100.
                "ranks": {c: [r[query].index(d) + 1 if d in r[query] else None for d in relevant]
                          for c, r in ranks.items()},
            })
    CASES.parent.mkdir(parents=True, exist_ok=True)
    with CASES.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    by_set = {}
    for case in cases:
        by_set[case["set"]] = by_set.get(case["set"], 0) + 1
    print(f"wrote {CASES.relative_to(ROOT)}: {len(cases)} cases {by_set}")
    return 0


def build(args) -> int:
    cases = [json.loads(line) for line in CASES.open(encoding="utf-8")]
    for source, spec in SETS.items():
        mine = [c for c in cases if c["source_dataset"] == source]
        if not mine:
            continue
        corpus_path = next((p for p in spec["corpus"] if p.is_file()), None)
        if corpus_path is None:
            hint = ("scripts/eval_suite.py prepare --suite cqadupstack-android" if source == "cqadupstack-android"
                    else "env/bin/python scripts/fetch_public_corpus.py datasets/public-mix.tsv --per-source 1000")
            raise SystemExit(f"{spec['name']}: no distractor corpus; make one with: {hint}")
        corpus = read_tsv(corpus_path)
        # The case documents are always present with their recorded text, also when a
        # corpus that was fetched again differs from the corpus of the original run.
        changed = 0
        for case in mine:
            for doc in case["relevant"]:
                changed += corpus.get(doc["doc_id"]) != doc["text"]
                corpus[doc["doc_id"]] = doc["text"]
        name = spec["name"]
        corpus_out = DATASETS_DIR / f"{name}.tsv"
        corpus_out.parent.mkdir(parents=True, exist_ok=True)
        with corpus_out.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f, delimiter="\t", lineterminator="\n").writerows(corpus.items())
        test_dir = TESTSET_DIR / name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / "questions.test.tsv").open("w", encoding="utf-8", newline="") as f:
            csv.writer(f, delimiter="\t", lineterminator="\n").writerows(
                (c["source_query_id"], c["query"]) for c in mine)
        qrels = {c["source_query_id"]: {d["doc_id"]: 1 for d in c["relevant"]} for c in mine}
        (test_dir / "qrels.test.json").write_text(json.dumps(qrels, separators=(",", ":")) + "\n")
        (test_dir / "collection_map.json").write_text("null\n")
        (test_dir / "dataset.json").write_text(json.dumps({
            "note": f"short-document regression cases ({len(mine)}) over {source}",
            "corpus": str(corpus_path.relative_to(ROOT)),
        }, indent=2) + "\n")
        print(f"{name}: {len(mine)} cases, {len(corpus)} documents from {corpus_path.relative_to(ROOT)}"
              + (f" ({changed} case documents added or restored)" if changed else ""))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select", help="select the cases from saved eval results")
    p.add_argument("--results", required=True, help="an eval_suite.py result directory")
    p.add_argument("--reference", default="mb-full-qsal", help="configuration that must find the documents")
    p.add_argument("--pruned", default="mb-default", help="configuration that must lose the documents")
    p.add_argument("--also", nargs="*", default=["mb-keep070-qsal"], help="more configurations to record ranks for")
    p.add_argument("--max-words", type=int, default=50, help="every relevant document has fewer words than this")
    sub.add_parser("build", help="write the regression datasets from cases.jsonl")
    args = parser.parse_args()
    return select(args) if args.command == "select" else build(args)


if __name__ == "__main__":
    sys.exit(main())

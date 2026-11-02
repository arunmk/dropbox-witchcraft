#!/usr/bin/env python3
"""Statistical Pickbrain test for the short-document pruning fix.

The old pruning default (keep 30% of document tokens) removed short documents
from semantic search. In Pickbrain, BM25 keyword search and the short result
list hide most of that failure, and the random seed of each index moves single
notes by many places. So this test measures a rate over many queries, not a
pass or a fail for single notes.

The test:
1. writes the public mixed corpus (Enron emails, GitHub files and Wikipedia
   pages) as Claude memory notes into a temporary Claude directory, with the
   recorded text of each short document from
   testset/regression/short-docs/pickbrain-queries.jsonl;
2. indexes the notes with Pickbrain in two databases: "before" with
   WARP_DOC_TOKEN_KEEP_FRACTION=0.3 (or a Pickbrain binary from master), and
   "after" with the default of the Pickbrain binary (0.7 on this branch);
3. runs each short-document query (420 queries, each with one relevant note of
   fewer than 50 words) in both databases, and reports how many relevant notes
   Pickbrain shows, how many are in the top 5, and how many notes only one of the
   two configurations shows;
4. repeats steps 2 and 3 for --builds builds, because each index has its own
   random seed.

Results from 2026-11-02 (two builds, 420 queries): Pickbrain showed 67.6% and
67.1% of the notes at keep 30%, and 72.9% and 71.4% at keep 70%, a gain of
+5.2 and +4.3 points. In each build, 33-38 notes were shown only at keep 70%,
and 15-16 notes only at keep 30%.

The test uses a temporary HOME and a temporary PICKBRAIN_DIR, so it never reads
your real Claude, Codex, Pi or Slack data, and never changes your real Pickbrain
database.

  make pickbrain
  env/bin/python scripts/pickbrain_short_doc_test.py --builds 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "testset" / "regression" / "short-docs" / "pickbrain-queries.jsonl"
CORPORA = [ROOT / "datasets" / "eval" / "public-mix-hard.tsv", ROOT / "datasets" / "public-mix.tsv"]
# Settings that add other sources or change the pruning; the test clears them.
CLEARED = ["EXTRA_CODEX_DIRS", "EXTRA_PI_DIRS", "PRE_INGEST_COMMAND",
           "WARP_DOC_TOKEN_KEEP_FRACTION", "WARP_DOC_TOKEN_MIN_KEEP"]
NOTE_PATH = re.compile(r"/memory/([^ \n]+\.md)")

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**63 else 2**31 - 1)


def note_name(doc_id: str) -> str:
    return doc_id.replace(":", "-") + ".md"


def write_notes(work: Path, queries: list[dict]) -> Path:
    corpus_path = next((p for p in CORPORA if p.is_file()), None)
    if corpus_path is None:
        raise SystemExit("no public mixed corpus; make one with:\n"
                         "  env/bin/python scripts/fetch_public_corpus.py datasets/public-mix.tsv --per-source 1000")
    with corpus_path.open(encoding="utf-8", newline="") as f:
        docs = {row[0]: row[1] for row in csv.reader(f, delimiter="\t") if len(row) >= 2}
    for q in queries:
        docs[q["doc_id"]] = q["text"]
    claude = work / "claude"
    notes = claude / "projects" / "-short-doc-test" / "memory"
    notes.mkdir(parents=True, exist_ok=True)
    for doc_id, body in docs.items():
        (notes / note_name(doc_id)).write_text(body + "\n", encoding="utf-8")
    print(f"wrote {len(docs)} memory notes from {corpus_path.relative_to(ROOT)}")
    return claude


class Pickbrain:
    def __init__(self, binary: str, work: Path, claude: Path, db: str, keep: str | None):
        self.binary, self.keep = binary, keep
        self.env = {k: v for k, v in os.environ.items() if k not in CLEARED}
        (work / "home").mkdir(exist_ok=True)
        self.env.update(HOME=str(work / "home"), EXTRA_CLAUDE_DIRS=str(claude), PICKBRAIN_DIR=str(work / db))
        if keep:
            self.env["WARP_DOC_TOKEN_KEEP_FRACTION"] = keep

    def ingest(self) -> None:
        subprocess.run([self.binary, "--ingest-only", "--quiet"], env=self.env, check=True)

    def rank(self, query: str, note: str) -> int | None:
        out = subprocess.run([self.binary, query], env=self.env, capture_output=True, text=True, check=True).stdout
        notes = NOTE_PATH.findall(out)
        return notes.index(note) + 1 if note in notes else None


def run_build(args, work: Path, claude: Path, queries: list[dict], build: int) -> dict:
    before_bin = args.before_bin or args.pickbrain
    configs = {
        "before": Pickbrain(before_bin, work, claude, f"build{build}-before", None if args.before_bin else "0.3"),
        "after": Pickbrain(args.pickbrain, work, claude, f"build{build}-after", None),
    }
    for name, pb in configs.items():
        print(f"build {build}: indexing the notes ({name})", flush=True)
        pb.ingest()
    jobs = [(name, q) for name in configs for q in queries]
    with ThreadPoolExecutor(args.jobs) as pool:
        ranks = list(pool.map(lambda job: configs[job[0]].rank(job[1]["query"], note_name(job[1]["doc_id"])), jobs))
    result = {"before": {}, "after": {}}
    for (name, q), r in zip(jobs, ranks):
        result[name][q["query_id"]] = r
    return result


def summarize(build: int, result: dict, n: int) -> dict:
    before, after = result["before"], result["after"]
    s = {
        "shown_before": sum(r is not None for r in before.values()),
        "shown_after": sum(r is not None for r in after.values()),
        "top5_before": sum(r is not None and r <= 5 for r in before.values()),
        "top5_after": sum(r is not None and r <= 5 for r in after.values()),
        "only_after": sum(after[q] is not None and before[q] is None for q in after),
        "only_before": sum(before[q] is not None and after[q] is None for q in after),
    }
    print(f"build {build}: notes shown before {s['shown_before']}/{n} ({s['shown_before'] / n:.1%}), "
          f"after {s['shown_after']}/{n} ({s['shown_after'] / n:.1%}); "
          f"top 5 before {s['top5_before']}, after {s['top5_after']}; "
          f"shown only after {s['only_after']}, only before {s['only_before']}")
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pickbrain", default=str(ROOT / "pickbrain"), help="the Pickbrain binary to test (default: ./pickbrain)")
    parser.add_argument("--before-bin", help="a Pickbrain binary for the 'before' side, for example one built "
                        "from master; without it, 'before' is --pickbrain with WARP_DOC_TOKEN_KEEP_FRACTION=0.3")
    parser.add_argument("--builds", type=int, default=1, help="number of index builds for each side (default: 1)")
    parser.add_argument("--jobs", type=int, default=3, help="parallel Pickbrain queries (default: 3)")
    parser.add_argument("--work", help="work directory (default: a new temporary directory)")
    parser.add_argument("--keep-work", action="store_true", help="keep the work directory at the end")
    parser.add_argument("--limit", type=int, help="use only the first N queries, for a quick check")
    args = parser.parse_args()

    for binary in filter(None, [args.pickbrain, args.before_bin]):
        if not os.access(binary, os.X_OK):
            raise SystemExit(f"Pickbrain binary not found: {binary}; run: make pickbrain")
    queries = [json.loads(line) for line in QUERIES.open(encoding="utf-8")][:args.limit]
    work = Path(args.work or tempfile.mkdtemp(prefix="pickbrain-short-doc-"))
    work.mkdir(parents=True, exist_ok=True)
    before = f"{args.before_bin} (its own default)" if args.before_bin else "keep 30% (WARP_DOC_TOKEN_KEEP_FRACTION=0.3)"
    print(f"work directory: {work}\nbefore: {before}\nafter:  {args.pickbrain} (its default)\n"
          f"{len(queries)} queries, each with one relevant note of fewer than 50 words")

    claude = write_notes(work, queries)
    results, summaries = [], []
    for build in range(1, args.builds + 1):
        result = run_build(args, work, claude, queries, build)
        results.append(result)
        summaries.append(summarize(build, result, len(queries)))
    (work / "results.json").write_text(json.dumps({"queries": len(queries), "builds": results}, indent=1))

    n = len(queries)
    gains = [s["shown_after"] - s["shown_before"] for s in summaries]
    print(f"\nsummary over {len(summaries)} build(s): after minus before = {gains} notes "
          f"({', '.join(f'{g / n:+.1%}' for g in gains)})")
    if all(g > 0 for g in gains):
        print("result: 'after' showed more short notes than 'before' in every build")
    elif all(g < 0 for g in gains):
        print("result: 'after' showed fewer short notes than 'before' in every build")
    else:
        print("result: the direction changed between builds; run more builds (--builds)")
    print(f"per-query ranks: {work / 'results.json'}")
    if not args.keep_work and not args.work:
        shutil.rmtree(work / "claude", ignore_errors=True)
        for build in range(1, args.builds + 1):
            for side in ("before", "after"):
                shutil.rmtree(work / f"build{build}-{side}", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

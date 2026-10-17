#!/usr/bin/env python3
"""Make a known-item search test set from any corpus with Claude.

Most personal file search is known-item search: the user looks for one file
that they saw before. This script samples documents from a corpus TSV
(id<TAB>body, the warp-cli readcsv format) and asks Claude for three queries
for each document:

  keyword   a few words, as typed in a search box
  question  a natural-language question that the document answers
  vague     a partial memory of the document, with different words

The source document is the only relevant result for each query. The output
uses the layout of scripts/eval_suite.py, so you can run:

  scripts/make_known_item_set.py corpus.tsv --name enron-ki --docs 300
  scripts/eval_suite.py run --suite enron-ki --configs ablation

CAUTION: generated queries often copy words from the document, which makes
BM25 look better than real use. The prompt asks for other words, and the
script reports the word overlap for each style. Use --max-overlap to make
harder queries: when a query copies too many words, the script sends the copied
words back to Claude and asks for a new query with other words. Do not tune
settings on this set alone.

  scripts/make_known_item_set.py corpus.tsv --name hard-ki --docs 300 \\
      --max-overlap keyword=0.5,question=0.4,vague=0.3 --rewrites 2

Use a public corpus that looks like a Dropbox account, for example the Enron
emails, public government PDFs and slides (converted to text), GitHub READMEs
and code, or Wikipedia pages. Do not use private user data.

Requires the anthropic package and credentials (ANTHROPIC_API_KEY or an
`ant auth login` profile). Each document costs one API request.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import random
import re
import shutil
import sys
import threading
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-5-5"
STYLES = ("keyword", "question", "vague")
# The model sees only this many characters of each document. Queries must come
# from this part, so a long document is still a fair target.
MAX_DOC_CHARS = 8000
MIN_DOC_WORDS = 20
# Prices in USD for each million tokens, for the cost line at the end of a run.
PRICE_INPUT, PRICE_OUTPUT = 4.0, 20.0
USAGE = {"requests": 0, "input": 0, "output": 0}
USAGE_LOCK = threading.Lock()

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**63 else 2**31 - 1)

SYSTEM = """You write search queries for a retrieval test set.
A user saw the given document some time ago and now wants to find it again in a
large personal collection of files (email, notes, reports, slides, code, PDFs).
Write the queries that this user would type. Each query must match this document
and must not be so general that many other documents also match.

Write exactly one query for each style:
- keyword: 2 to 5 words, as typed into a search box. No full sentence.
- question: one natural-language question that this document answers.
- vague: what the user remembers about the document, in their own words.
  It can be incomplete. Use words that are different from the document's words.

For all styles, prefer paraphrase to words copied from the document. Do not
copy long phrases. Use exact names, numbers or identifiers only if a user would
remember them. Write the queries in the language of the document."""

SCHEMA = {
    "type": "object",
    "properties": {style: {"type": "string"} for style in STYLES},
    "required": list(STYLES),
    "additionalProperties": False,
}

WORD = re.compile(r"\w+", re.UNICODE)
STOPWORDS = set("""a an and are as at be by for from has have how i in is it its of on or
that the this to was were what when where which who why will with you your my me
do does did about can could should would""".split())


def content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1}


def overlap(query: str, document: str) -> float:
    """The fraction of the query's content words that also occur in the document."""
    words = content_words(query)
    return len(words & content_words(document)) / len(words) if words else 0.0


def read_corpus(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [(row[0], row[1]) for row in csv.reader(f, delimiter="\t") if len(row) >= 2]


REWRITE = """The queries below copy too many words from the document. A real user
rarely remembers the exact words of a document. Write each listed query again,
in the same style, so that the query still matches only this document.
Do not use the listed copied words. Use synonyms, more general words, or a
description of the purpose or the context of the document instead."""


def ask(client: anthropic.Anthropic, doc_id: str, content: str, styles: list[str],
        system: str) -> dict[str, str] | None:
    schema = {**SCHEMA, "properties": {s: {"type": "string"} for s in styles}, "required": styles}
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=2000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=system,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": content}],
    )
    with USAGE_LOCK:
        USAGE["requests"] += 1
        USAGE["input"] += response.usage.input_tokens
        USAGE["output"] += response.usage.output_tokens
    if response.stop_reason == "refusal":
        print(f"{doc_id}: request declined", file=sys.stderr)
        return None
    text = next((block.text for block in response.content if block.type == "text"), "")
    try:
        queries = json.loads(text)
    except json.JSONDecodeError:
        print(f"{doc_id}: response is not valid JSON", file=sys.stderr)
        return None
    return {s: " ".join(queries[s].split()) for s in styles if queries.get(s, "").strip()}


def generate(client: anthropic.Anthropic, doc_id: str, body: str,
             max_overlap: dict[str, float], rewrites: int) -> dict[str, dict] | None:
    """Return {style: {"query", "overlap", "rewrites"}} for one document."""
    document = f"<document>\n{body[:MAX_DOC_CHARS]}\n</document>"
    queries = ask(client, doc_id, document, list(STYLES), SYSTEM)
    if not queries:
        return None
    out = {style: {"query": q, "overlap": overlap(q, body), "rewrites": 0} for style, q in queries.items()}
    for attempt in range(1, rewrites + 1):
        over = [s for s, r in out.items() if s in max_overlap and r["overlap"] > max_overlap[s]]
        if not over:
            break
        doc_words = content_words(body)
        listed = "\n".join(
            f"- {s}: {out[s]['query']}\n  copied words: {', '.join(sorted(content_words(out[s]['query']) & doc_words))}"
            for s in over)
        better = ask(client, doc_id, f"{document}\n<queries>\n{listed}\n</queries>", over,
                     f"{SYSTEM}\n\n{REWRITE}")
        for s, q in (better or {}).items():
            score = overlap(q, body)
            if score < out[s]["overlap"]:
                out[s] = {"query": q, "overlap": score, "rewrites": attempt}
    return out


def parse_limits(spec: str | None) -> dict[str, float]:
    limits = {}
    for item in (spec or "").split(","):
        if item.strip():
            style, _, value = item.partition("=")
            if style.strip() not in STYLES:
                raise SystemExit(f"--max-overlap: unknown style {style!r}; use {', '.join(STYLES)}")
            limits[style.strip()] = float(value)
    return limits


def write_split(test_dir: Path, split: str, rows: list[tuple[str, str, str, str]]) -> None:
    with (test_dir / f"questions.{split}.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for query_id, _doc_id, _style, query in rows:
            writer.writerow([query_id, query])
    qrels = {query_id: {doc_id: 1} for query_id, doc_id, _style, _query in rows}
    (test_dir / f"qrels.{split}.json").write_text(json.dumps(qrels, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path, help="corpus TSV: id<TAB>body")
    parser.add_argument("--name", required=True, help="dataset name for eval_suite.py")
    parser.add_argument("--docs", type=int, default=200, help="documents to sample (default: 200)")
    parser.add_argument("--dev-fraction", type=float, default=0.5,
                        help="fraction of sampled documents whose queries go to the dev split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8, help="parallel API requests")
    parser.add_argument("--max-overlap", help="overlap limit for each style, for example "
                        "keyword=0.5,question=0.4,vague=0.3 (default: no limit)")
    parser.add_argument("--rewrites", type=int, default=2,
                        help="rewrite requests for a query above its --max-overlap limit (default: 2)")
    parser.add_argument("--datasets-dir", type=Path, default=ROOT / "datasets" / "eval")
    parser.add_argument("--testset-dir", type=Path, default=ROOT / "testset" / "eval")
    args = parser.parse_args()

    corpus = read_corpus(args.corpus)
    candidates = [(i, b) for i, b in corpus if len(b.split()) >= MIN_DOC_WORDS]
    rng = random.Random(args.seed)
    sample = rng.sample(candidates, min(args.docs, len(candidates)))
    print(f"{len(corpus)} documents, {len(candidates)} long enough, {len(sample)} sampled")

    limits = parse_limits(args.max_overlap)
    client = anthropic.Anthropic()
    results: dict[str, dict[str, dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(generate, client, doc_id, body, limits, args.rewrites): doc_id
                   for doc_id, body in sample}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            doc_id = futures[future]
            try:
                queries = future.result()
            except anthropic.APIError as e:
                print(f"{doc_id}: API error, document skipped: {e}", file=sys.stderr)
                continue
            if queries:
                results[doc_id] = queries
            if done % 25 == 0:
                print(f"{done}/{len(sample)} documents")

    bodies = dict(corpus)
    doc_ids = [doc_id for doc_id, _ in sample if doc_id in results]
    dev_count = round(len(doc_ids) * args.dev_fraction)
    splits = {"dev": doc_ids[:dev_count], "test": doc_ids[dev_count:]}

    test_dir = args.testset_dir / args.name
    test_dir.mkdir(parents=True, exist_ok=True)
    args.datasets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.corpus, args.datasets_dir / f"{args.name}.tsv")
    (test_dir / "collection_map.json").write_text("null\n")

    records = []
    for split, ids in splits.items():
        rows = []
        for doc_id in ids:
            for style, result in results[doc_id].items():
                query_id = f"{doc_id}:{style}"
                rows.append((query_id, doc_id, style, result["query"]))
                records.append({"split": split, "query_id": query_id, "doc_id": doc_id, "style": style,
                                "query": result["query"], "overlap": result["overlap"],
                                "rewrites": result["rewrites"],
                                "over_limit": result["overlap"] > limits.get(style, 1.0)})
        write_split(test_dir, split, rows)
    with (test_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (test_dir / "dataset.json").write_text(json.dumps({
        "note": f"known-item queries made by {MODEL} from {args.corpus.name}",
        "source": str(args.corpus), "docs_sampled": len(sample), "docs_with_queries": len(doc_ids),
        "max_overlap": limits, "rewrites": args.rewrites,
    }, indent=2) + "\n")

    print(f"wrote {test_dir} ({len(records)} queries from {len(doc_ids)} documents)")
    cost = (USAGE["input"] * PRICE_INPUT + USAGE["output"] * PRICE_OUTPUT) / 1e6
    print(f"API use: {USAGE['requests']} requests, {USAGE['input']} input tokens, "
          f"{USAGE['output']} output tokens, approximately ${cost:.2f} "
          f"(${cost / max(1, len(sample)):.3f} for each document)")
    print("mean word overlap with the source document (lower is harder and more realistic):")
    for style in STYLES:
        values = [r["overlap"] for r in records if r["style"] == style]
        if values:
            print(f"  {style:8s} {sum(values) / len(values):.2f}")
    if limits:
        over = sum(r["over_limit"] for r in records)
        print(f"{over} of {len(records)} queries are still above their limit after {args.rewrites} rewrites")
    return 0


if __name__ == "__main__":
    sys.exit(main())

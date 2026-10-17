#!/usr/bin/env python3
"""Fetch a public corpus that looks like the files in a personal Dropbox account.

The corpus mixes three public sources, with random documents from each:
  enron  emails and memos (corbt/enron-emails), many of them short
  code   README files of random GitHub repositories (GitHub search API) and
         Python source files (codeparrot/codeparrot-clean-valid)
  wiki   English Wikipedia pages (wikimedia/wikipedia, 20231101.en), many of them long

The script downloads one Parquet shard of each Hugging Face dataset (cached in
.cache/public-corpus) and README files from GitHub. It needs pyarrow. The same --seed gives the same corpus. Output: a TSV with
id<TAB>body, for scripts/make_known_item_set.py and warp-cli readcsv.

  scripts/fetch_public_corpus.py datasets/public-mix.tsv --per-source 1000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co/api/datasets"
CACHE = Path(__file__).resolve().parents[1] / ".cache" / "public-corpus"
# Very long pages make indexing slow and do not add much. Cut them at this size.
MAX_CHARS = 30_000
MIN_WORDS = 5

GITHUB_LANGUAGES = ["python", "javascript", "go", "java", "rust", "c++", "shell", "ruby", "php", "typescript"]
STAR_RANGES = ["10..50", "50..200", "200..1000", "1000..5000"]


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "witchcraft-eval-corpus"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == 5:
                raise
        except OSError:
            if attempt == 5:
                raise
        time.sleep(2 ** attempt)  # Servers return 5xx and 429 under load; wait and retry.
    raise AssertionError("unreachable")


def get_json(url: str, params: dict | None = None) -> dict:
    return json.loads(fetch(f"{url}?{urllib.parse.urlencode(params)}" if params else url))


def sample_rows(dataset: str, config: str, count: int, rng: random.Random, columns: list[str]):
    """Download one random Parquet shard (cached) and return random rows from it."""
    import pyarrow.parquet as pq  # Only this script needs pyarrow.

    shards = get_json(f"{HUB}/{dataset}/parquet/{config}/train")
    url = rng.choice(shards)
    path = CACHE / dataset.replace("/", "--") / config / url.rsplit("/", 1)[1]
    if not path.is_file():
        print(f"downloading {url}", file=sys.stderr)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(fetch(url))
        tmp.replace(path)
    table = pq.read_table(path, columns=columns)
    for index in rng.sample(range(table.num_rows), min(count, table.num_rows)):
        yield {name: table.column(name)[index].as_py() for name in columns}


def clean(text: str) -> str:
    return " ".join(text.split())[:MAX_CHARS]


def enron(count: int, rng: random.Random):
    columns = ["message_id", "subject", "from", "date", "body"]
    for row in sample_rows("corbt/enron-emails", "default", count * 2, rng, columns):
        subject = row.get("subject") or ""
        body = f"Subject: {subject} From: {row.get('from', '')} Date: {row.get('date', '')} {row.get('body', '')}"
        yield f"enron:{hashlib.sha1(row['message_id'].encode()).hexdigest()[:16]}", clean(body)


def github_readmes(count: int, rng: random.Random):
    repos = []
    for language in GITHUB_LANGUAGES:
        stars = rng.choice(STAR_RANGES)
        query = {"q": f"stars:{stars} language:{language}", "per_page": 100, "page": rng.randint(1, 5)}
        repos += [r["full_name"] for r in get_json("https://api.github.com/search/repositories", query)["items"]]
        time.sleep(7)  # Search without login allows 10 requests a minute.
    rng.shuffle(repos)
    found = 0
    for repo in repos:
        if found >= count:
            return
        for name in ("README.md", "readme.md", "README.rst", "README"):
            try:
                text = fetch(f"https://raw.githubusercontent.com/{repo}/HEAD/{name}").decode("utf-8", "replace")
            except Exception:
                continue
            found += 1
            yield f"code:{hashlib.sha1(repo.encode()).hexdigest()[:16]}", clean(f"{repo} {name} {text}")
            break


def python_files(count: int, rng: random.Random):
    for row in sample_rows("codeparrot/codeparrot-clean-valid", "default", count * 2, rng,
                           ["repo_name", "path", "content"]):
        key = f"{row['repo_name']}/{row['path']}"
        yield f"code:{hashlib.sha1(key.encode()).hexdigest()[:16]}", clean(f"{row['path']} {row['content']}")


def code(count: int, rng: random.Random):
    yield from github_readmes(count // 2 + count // 4, rng)
    yield from python_files(count - count // 2, rng)


def wiki(count: int, rng: random.Random):
    for row in sample_rows("wikimedia/wikipedia", "20231101.en", count * 2, rng, ["id", "title", "text"]):
        yield f"wiki:{row['id']}", clean(f"{row['title']} {row['text']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-source", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    docs: list[tuple[str, str]] = []
    for name, source in [("enron", enron), ("code", code), ("wiki", wiki)]:
        seen_ids, seen_bodies, kept = set(), set(), []
        for doc_id, body in source(args.per_source, rng):
            digest = hashlib.sha1(body.encode()).digest()
            # Enron has many copies of the same mail; keep one copy.
            if doc_id in seen_ids or digest in seen_bodies or len(body.split()) < MIN_WORDS:
                continue
            seen_ids.add(doc_id)
            seen_bodies.add(digest)
            kept.append((doc_id, body))
        rng.shuffle(kept)
        kept = kept[:args.per_source]
        print(f"{name}: {len(kept)} documents")
        docs += kept

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(docs)
    print(f"wrote {args.output} ({len(docs)} documents)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

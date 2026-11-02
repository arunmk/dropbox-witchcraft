#!/usr/bin/env python3
"""Retrieval evaluation suite: datasets, ablations, cost metrics and reports.

Subcommands:
  prepare   Download and convert the datasets of a suite.
  run       Build the encoders, index each dataset, run each configuration,
            and write results.json and report.md.
  report    Write report.md again from an existing results.json.
  pipeline  Run the ablation, the dev sweeps and the pruning check in one
            command, with one shared index directory, and write summary.md.
  compare   Compare the same configurations in two result directories, for
            example master against a branch.

Examples:
  scripts/eval_suite.py prepare --suite small
  scripts/eval_suite.py run --suite small --configs ablation
  scripts/eval_suite.py pipeline --plan standard
  scripts/eval_suite.py run --suite smoke --configs sweep --split dev
  scripts/eval_suite.py run --suite smoke --configs default \\
      --check 'nfcorpus:ndcg@10>=0.31'
  scripts/eval_suite.py compare bench-results/eval/master bench-results/eval/branch

Only the Python standard library is used. Metrics come from tools/trec-score.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets" / "eval"
TESTSET_DIR = ROOT / "testset" / "eval"
DEFAULT_OUT = ROOT / "bench-results" / "eval"
SCORE_BIN = ROOT / "target" / "release" / ("trec-score.exe" if os.name == "nt" else "trec-score")

# Print progress at once, also when the output goes to a log file.
sys.stdout.reconfigure(line_buffering=True)
csv.field_size_limit(sys.maxsize if sys.maxsize < 2**63 else 2**31 - 1)

# BEIR datasets. The set covers different domains, document lengths and query
# styles, because the target content (general Dropbox files) is not known.
# max_docs keeps every judged document and adds random distractors; such a
# corpus is easier than the full corpus, so compare it only with itself.
DATASETS = {
    "nfcorpus": dict(beir="nfcorpus", note="medical, keyword-like queries"),
    "scifact": dict(beir="scifact", note="science claims, long abstracts"),
    "arguana": dict(beir="arguana", note="arguments, query is a full passage"),
    "scidocs": dict(beir="scidocs", note="paper titles to abstracts"),
    "fiqa": dict(beir="fiqa", note="finance questions, forum answers"),
    "cqadupstack-android": dict(beir="cqadupstack/android", note="technical Q&A, short titles"),
    "trec-covid": dict(beir="trec-covid", note="medical, pooled qrels"),
    "touche": dict(beir="webis-touche2020", note="arguments, long web text"),
    "quora-100k": dict(beir="quora", max_docs=100_000, max_queries=1000,
                       note="short questions, duplicates (sampled)"),
    "nq-100k": dict(beir="nq", max_docs=100_000, max_queries=1000,
                    note="Wikipedia, natural questions (sampled)"),
    # Regression sets: queries whose short relevant documents pruning once removed from the
    # top 100. scripts/short_doc_regression.py builds them; see testset/regression/short-docs/.
    "regress-short-cqa": dict(builder="short_doc_regression.py",
                              note="regression: short CQADupStack questions lost by pruning"),
    "regress-short-mix": dict(builder="short_doc_regression.py",
                              note="regression: short emails and Wikipedia stubs lost by pruning"),
}

SUITES = {
    "smoke": ["nfcorpus"],
    "small": ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa", "cqadupstack-android"],
}
SUITES["full"] = SUITES["small"] + ["trec-covid", "touche", "quora-100k", "nq-100k"]
SUITES["regression"] = ["regress-short-cqa", "regress-short-mix"]

MODERNBERT = "modernbert-quantized"
T5 = "t5-quantized"


@dataclass(frozen=True)
class Config:
    name: str
    encoder: str | None  # None means full text only, with no embeddings.
    keep: float | None = None  # WARP_DOC_TOKEN_KEEP_FRACTION; None for encoders without a gate.
    query_salience: bool | None = None
    mode: str = "querycsv"
    description: str = ""
    min_keep: int = 0  # WARP_DOC_TOKEN_MIN_KEEP: the minimum tokens kept in each document.

    @property
    def build_key(self) -> str:
        if self.encoder is None:
            return "fulltext"
        if self.keep is None:
            return self.encoder
        key = f"{self.encoder}-keep{round(self.keep * 100):03d}"
        return f"{key}-min{self.min_keep}" if self.min_keep else key

    def env(self) -> dict[str, str]:
        env = {}
        if self.keep is not None:
            env["WARP_DOC_TOKEN_KEEP_FRACTION"] = repr(self.keep)
            env["WARP_DOC_TOKEN_MIN_KEEP"] = str(self.min_keep)
        if self.query_salience is not None:
            env["WARP_QUERY_TOKEN_SALIENCE"] = "1" if self.query_salience else "0"
        return env


def modernbert(keep, salience, mode="querycsv", name=None, description=""):
    name = name or f"mb-keep{round(keep * 100):03d}{'-qsal' if salience else ''}" + (
        "-hybrid" if mode == "hybridcsv" else "")
    return Config(name, MODERNBERT, keep, salience, mode, description)


PRESETS = {
    "bm25": Config("bm25", None, mode="fulltextcsv", description="SQLite FTS5 BM25 only"),
    "t5": Config("t5", T5, description="XTR T5 encoder, no gate"),
    "mb-full": modernbert(1.0, False, name="mb-full", description="ModernBERT, no pruning, no salience"),
    "mb-full-qsal": modernbert(1.0, True, name="mb-full-qsal",
                               description="ModernBERT, no pruning, query salience on"),
    "mb-prune": modernbert(0.3, False, name="mb-prune", description="+ keep 30% of document tokens"),
    "mb-default": modernbert(0.3, True, name="mb-default",
                             description="+ query salience (keep 30%: the default until 2026-10-17)"),
    "mb-default-hybrid": modernbert(0.3, True, "hybridcsv", name="mb-default-hybrid",
                                    description="keep 30% + query salience + BM25 fusion"),
    "mb-keep050-qsal": modernbert(0.5, True, description="keep 50% of document tokens, query salience on"),
    "mb-keep070-qsal": modernbert(0.7, True,
                                  description="keep 70% of document tokens, query salience on (the current default)"),
    "mb-keep070": modernbert(0.7, False, description="keep 70% of document tokens, query salience off"),
    "mb-keep030-min32-qsal": Config("mb-keep030-min32-qsal", MODERNBERT, 0.3, True, min_keep=32,
                                    description="keep 30%, but at least 32 tokens for each document"),
}

CONFIG_SETS = {
    # "default" is the current default configuration (keep 70% since 2026-10-17). The preset
    # "mb-default" keeps its name and its 30% value, so that older result directories still compare.
    "default": ["mb-keep070-qsal"],
    "ablation": ["bm25", "t5", "mb-full", "mb-full-qsal", "mb-prune", "mb-default", "mb-default-hybrid",
                 "mb-keep070-qsal"],
    "ablation-nomodel": ["bm25", "mb-full", "mb-full-qsal", "mb-prune", "mb-default", "mb-default-hybrid"],
    "pruning-check": ["mb-full-qsal", "mb-default", "mb-keep050-qsal", "mb-keep030-min32-qsal"],
    "keep70-check": ["mb-full", "mb-full-qsal", "mb-default", "mb-keep070", "mb-keep070-qsal"],
}

# Ablation steps: each pair changes one thing. The report shows each pair whose
# two configurations are in the run.
STEPS = [
    ("bm25", "t5", "a learned encoder (T5) instead of BM25"),
    ("t5", "mb-full", "ModernBERT instead of T5"),
    ("mb-full", "mb-prune", "pruning, without query salience"),
    ("mb-full-qsal", "mb-default", "pruning, with query salience"),
    ("mb-full", "mb-full-qsal", "query salience, without pruning"),
    ("mb-prune", "mb-default", "query salience, with pruning"),
    ("mb-full", "mb-default", "pruning and query salience together"),
    ("mb-default", "mb-default-hybrid", "BM25 fusion"),
    ("mb-default", "mb-keep050-qsal", "keep 50% instead of 30%"),
    ("mb-default", "mb-keep030-min32-qsal", "a minimum of 32 tokens for each document"),
    ("mb-full-qsal", "mb-keep050-qsal", "keep 50% instead of 100%"),
    ("mb-full-qsal", "mb-keep030-min32-qsal", "keep 30% with a minimum of 32, instead of 100%"),
    ("mb-default", "mb-keep070-qsal", "keep 70% instead of 30%"),
    ("mb-full-qsal", "mb-keep070-qsal", "keep 70% instead of 100%, with query salience"),
    ("mb-full", "mb-keep070", "keep 70% instead of 100%, without query salience"),
    ("mb-keep070", "mb-keep070-qsal", "query salience, at keep 70%"),
]


def comparison_steps(configs: list[str]) -> list[tuple[str, str, str]]:
    steps = [step for step in STEPS if step[0] in configs and step[1] in configs]
    # A sweep compares each keep fraction with the run that keeps all tokens.
    sweep = [c for c in configs if re.fullmatch(r"mb-keep\d{3}-qsal", c)]
    if "mb-keep100-qsal" in sweep:
        steps += [("mb-keep100-qsal", c, f"keep {int(c[7:10])}% instead of 100%")
                  for c in sweep if c != "mb-keep100-qsal"]
    # A minimum sweep compares each minimum with plain 30% pruning.
    plain = next((c for c in ("mb-keep030-qsal", "mb-default") if c in configs), None)
    if plain:
        steps += [(plain, c, f"a minimum of {re.search(r'min(\d+)', c).group(1)} tokens for each document")
                  for c in configs if re.fullmatch(r"mb-keep030-min\d+-qsal", c)
                  and (plain, c) not in {(a, b) for a, b, _ in steps}]
    if not steps:
        steps = [(a, b, f"{a} to {b}") for a, b in zip(configs, configs[1:])]
    return steps


def min_sweep_configs(mins: list[int]) -> list[Config]:
    return [Config(f"mb-keep030-min{m}-qsal", MODERNBERT, 0.3, True, min_keep=m,
                   description=f"keep 30%, but at least {m} tokens for each document")
            for m in mins if m > 0]


def sweep_configs(keeps: list[float]) -> list[Config]:
    return [modernbert(k, True, description=f"keep {k:.0%} of document tokens, query salience on")
            for k in keeps]


def resolve_configs(spec: str, keeps: list[float], mins: list[int] = (16, 32, 64)) -> list[Config]:
    configs: list[Config] = []
    for item in spec.split(","):
        item = item.strip()
        if item == "sweep":
            configs += sweep_configs(keeps)
        elif item == "min-sweep":
            configs += min_sweep_configs(list(mins))
        elif item in CONFIG_SETS:
            configs += [PRESETS[name] for name in CONFIG_SETS[item]]
        elif item in PRESETS:
            configs.append(PRESETS[item])
        else:
            raise SystemExit(f"unknown config or config set: {item}; "
                             f"known: {sorted(PRESETS) + sorted(CONFIG_SETS) + ['sweep', 'min-sweep']}")
    seen, unique = set(), []
    for config in configs:
        if config.name not in seen:
            seen.add(config.name)
            unique.append(config)
    return unique


def resolve_datasets(spec: str) -> list[str]:
    names: list[str] = []
    for item in spec.split(","):
        item = item.strip()
        for name in SUITES.get(item, [item]):
            if name not in DATASETS and not (TESTSET_DIR / name).is_dir():
                raise SystemExit(f"unknown dataset: {name}; known: {sorted(DATASETS)}, "
                                 f"or a directory in {TESTSET_DIR}")
            if name not in names:
                names.append(name)
    return names


# --- dataset preparation -----------------------------------------------------

def dataset_files(name: str, split: str) -> dict[str, Path]:
    test_dir = TESTSET_DIR / name
    return {
        "corpus": DATASETS_DIR / f"{name}.tsv",
        "questions": test_dir / f"questions.{split}.tsv",
        "qrels": test_dir / f"qrels.{split}.json",
        "map": test_dir / "collection_map.json",
    }


def dataset_note(name: str) -> str:
    if name in DATASETS:
        return DATASETS[name]["note"]
    info = TESTSET_DIR / name / "dataset.json"
    return json.loads(info.read_text()).get("note", "custom") if info.is_file() else "custom"


def known_item_records(name: str, split: str) -> list[dict]:
    """Query records from make_known_item_set.py, if the dataset has them."""
    path = TESTSET_DIR / name / "queries.jsonl"
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    return [r for r in records if r["split"] == split]


def query_styles(name: str, split: str) -> dict[str, str]:
    return {r["query_id"]: r["style"] for r in known_item_records(name, split)}


# The fraction of the query's content words that also occur in the source document.
# BM25 is strong when this overlap is high, so the report shows each bucket.
OVERLAP_BUCKETS = [("overlap <0.4", 0.0, 0.4), ("overlap 0.4-0.6", 0.4, 0.6),
                   ("overlap 0.6-0.8", 0.6, 0.8), ("overlap >=0.8", 0.8, None)]


def query_overlaps(name: str, split: str) -> dict[str, str]:
    out = {}
    for r in known_item_records(name, split):
        for label, low, high in OVERLAP_BUCKETS:
            if r["overlap"] >= low and (high is None or r["overlap"] < high):
                out[r["query_id"]] = label
    return out


def prepare(names: list[str], cache_dir: Path) -> None:
    for name in names:
        files = dataset_files(name, "test")
        if files["corpus"].is_file() and files["qrels"].is_file():
            print(f"{name}: already prepared")
            continue
        if name not in DATASETS:
            raise SystemExit(f"{name}: custom dataset files are missing in {TESTSET_DIR / name}")
        spec = DATASETS[name]
        if "builder" in spec:
            # A regression set needs the corpus of its source dataset.
            if name == "regress-short-cqa":
                prepare(["cqadupstack-android"], cache_dir)
            print(f"{name}: building with scripts/{spec['builder']}")
            subprocess.run([sys.executable, str(ROOT / "scripts" / spec["builder"]), "build"], cwd=ROOT, check=True)
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "prepare_beir_dataset.py"), spec["beir"],
               "--output-name", name, "--cache-dir", str(cache_dir),
               "--datasets-dir", str(DATASETS_DIR), "--testset-dir", str(TESTSET_DIR)]
        if "max_docs" in spec:
            cmd += ["--max-docs", str(spec["max_docs"])]
        if "max_queries" in spec:
            cmd += ["--max-queries", str(spec["max_queries"])]
        print(f"{name}: preparing from BEIR {spec['beir']}")
        subprocess.run(cmd, cwd=ROOT, check=True)


# --- process execution -------------------------------------------------------

@dataclass
class StepResult:
    seconds: float
    peak_rss_mb: float | None
    stdout: str


def run_step(cmd: list[str], cwd: Path, env: dict[str, str], log: Path) -> StepResult:
    merged = os.environ.copy()
    merged.update(env)
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n# env {env}\n")
        log_file.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, env=merged, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace")
        lines = []
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            log_file.write(line)
        peak = None
        if hasattr(os, "wait4"):
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            # macOS reports bytes and Linux reports kilobytes.
            scale = 1 / (1024 * 1024) if sys.platform == "darwin" else 1 / 1024
            peak = usage.ru_maxrss * scale
        else:
            proc.wait()
    seconds = time.monotonic() - start
    if proc.returncode != 0:
        tail = "".join(lines[-20:])
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\nsee {log}\n{tail}")
    return StepResult(seconds, peak, "".join(lines))


def build_cli(encoder: str, bin_dir: Path, log_dir: Path) -> Path:
    target = bin_dir / f"warp-cli-{encoder}"
    if target.is_file():
        return target
    print(f"building warp-cli for {encoder}")
    run_step(["make", "warp-cli", f"ENCODER={encoder}", "EXTRA_FEATURES=deterministic"],
             ROOT, {}, log_dir / f"build-{encoder}.log")
    bin_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2((ROOT / "warp-cli").resolve(), target)
    return target


def build_scorer(log_dir: Path) -> None:
    run_step(["cargo", "build", "--release", "-p", "trec-score"], ROOT, {}, log_dir / "build-trec-score.log")


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def parse_latency(stdout: str) -> dict[str, float]:
    out = {}
    for name, key in [("p50 total search latency", "p50_ms"), ("p95 total search latency", "p95_ms"),
                      ("p50 embedder latency", "p50_embed_ms"), ("p95 embedder latency", "p95_embed_ms")]:
        match = re.search(rf"{name} = (\d+) ms", stdout)
        if match:
            out[key] = float(match.group(1))
    return out


def parse_metrics(stdout: str) -> dict[str, float]:
    out = {}
    for line in stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def read_per_query(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        return {row[0]: dict(zip(header[1:], map(float, row[1:]))) for row in reader}


def compare_files(base: Path, candidate: Path) -> dict[str, dict[str, float]]:
    out = subprocess.run([str(SCORE_BIN), "compare", str(base), str(candidate)],
                         check=True, capture_output=True, text=True).stdout
    rows = list(csv.reader(out.splitlines(), delimiter="\t"))
    header = rows[0]
    return {row[0]: {k: float(v) for k, v in zip(header[1:], row[1:])} for row in rows[1:]}


# --- length slices -----------------------------------------------------------

LENGTH_BUCKETS = [("short (<50 words)", 0, 50), ("medium (50-200)", 50, 200), ("long (>200)", 200, None)]


def length_slices(corpus: Path, qrels_path: Path) -> dict[str, str]:
    """Map each query to a bucket from the mean word count of its relevant documents."""
    qrels = json.loads(qrels_path.read_text())
    wanted = {doc for docs in qrels.values() for doc, grade in docs.items() if grade > 0}
    lengths = {}
    with corpus.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row and row[0] in wanted:
                lengths[row[0]] = len(row[1].split())
    out = {}
    for query, docs in qrels.items():
        values = [lengths[d] for d, g in docs.items() if g > 0 and d in lengths]
        if not values:
            continue
        mean = sum(values) / len(values)
        for label, low, high in LENGTH_BUCKETS:
            if mean >= low and (high is None or mean < high):
                out[query] = label
    return out


# --- the run -----------------------------------------------------------------

@dataclass
class BuildCost:
    docs: int
    readcsv_s: float
    embed_s: float
    index_s: float
    index_mb: float
    peak_rss_mb: float | None
    docs_per_s: float


@dataclass
class ConfigResult:
    metrics: dict[str, float]
    latency: dict[str, float]
    query_peak_rss_mb: float | None
    per_query_file: str
    slices: dict[str, dict[str, float]] = field(default_factory=dict)
    styles: dict[str, dict[str, float]] = field(default_factory=dict)
    overlap: dict[str, dict[str, float]] = field(default_factory=dict)


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def build_index(dataset: str, config: Config, cli: Path | None, run_dir: Path, rebuild: bool) -> BuildCost:
    cost_file = run_dir / "build.json"
    if cost_file.is_file() and not rebuild:
        return BuildCost(**json.loads(cost_file.read_text()))
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    files = dataset_files(dataset, "test")
    env = {"WARP_DB": str(run_dir / "mydb.sqlite"), "WARP_ASSETS": str(ROOT / "assets"), **config.env()}
    assert cli is not None
    logs = run_dir / "logs"
    print(f"  [{dataset}] build {config.build_key}: readcsv")
    readcsv = run_step([str(cli), "readcsv", str(files["corpus"])], run_dir, env, logs / "readcsv.log")
    embed = index = StepResult(0.0, None, "")
    if config.encoder is not None:
        print(f"  [{dataset}] build {config.build_key}: embed")
        embed = run_step([str(cli), "embed"], run_dir, env, logs / "embed.log")
        print(f"  [{dataset}] build {config.build_key}: index")
        index = run_step([str(cli), "index"], run_dir, env, logs / "index.log")
    docs = count_lines(files["corpus"])
    peaks = [s.peak_rss_mb for s in (embed, index) if s.peak_rss_mb is not None]
    work = embed.seconds + index.seconds
    cost = BuildCost(
        docs=docs, readcsv_s=readcsv.seconds, embed_s=embed.seconds, index_s=index.seconds,
        index_mb=directory_bytes(run_dir) / 1e6 - directory_bytes(logs) / 1e6,
        peak_rss_mb=max(peaks) if peaks else None,
        docs_per_s=docs / work if work > 0 else 0.0,
    )
    cost_file.write_text(json.dumps(asdict(cost), indent=2))
    return cost


def slice_means(rows: dict[str, dict[str, float]], labels: dict[str, str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in dict.fromkeys(labels.values()):
        chosen = [m for q, m in rows.items() if labels.get(q) == label]
        if chosen:
            out[label] = {"queries": len(chosen)}
            for metric in ["ndcg@10", "recall@100", "mrr@10", "success@1"]:
                out[label][metric] = sum(m[metric] for m in chosen) / len(chosen)
    return out


def run_config(dataset: str, split: str, config: Config, cli: Path, build_dir: Path,
               out_dir: Path, slices: dict[str, str]) -> ConfigResult:
    files = dataset_files(dataset, split)
    result_dir = out_dir / "runs" / dataset / split / config.name
    result_dir.mkdir(parents=True, exist_ok=True)
    env = {"WARP_DB": str(build_dir / "mydb.sqlite"), "WARP_ASSETS": str(ROOT / "assets"), **config.env()}
    results = result_dir / "results.tsv"
    print(f"  [{dataset}] {split} queries: {config.name}")
    query = run_step([str(cli), config.mode, str(files["questions"]), str(results)],
                     build_dir, env, result_dir / "query.log")
    per_query = result_dir / "per_query.tsv"
    scored = subprocess.run([str(SCORE_BIN), "metrics", str(results), str(files["map"]),
                             str(files["qrels"]), "--per-query", str(per_query)],
                            check=True, capture_output=True, text=True)
    rows = read_per_query(per_query)
    return ConfigResult(parse_metrics(scored.stdout), parse_latency(query.stdout), query.peak_rss_mb,
                        str(per_query.relative_to(out_dir)), slice_means(rows, slices),
                        slice_means(rows, query_styles(dataset, split)),
                        slice_means(rows, query_overlaps(dataset, split)))


def git_state() -> dict[str, str]:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": "yes" if git("status", "--porcelain", "-uno") else "no"}


def run(args) -> int:
    datasets = resolve_datasets(args.suite)
    configs = resolve_configs(args.configs, [float(k) for k in args.keeps.split(",")],
                              [int(m) for m in args.mins.split(",")])
    out_dir = Path(args.out or DEFAULT_OUT / dt.datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    work_dir = Path(args.work_dir).resolve() if args.work_dir else out_dir / "indexes"

    for name in datasets:
        if not dataset_files(name, args.split)["qrels"].is_file():
            if args.split == "test":
                prepare([name], Path(args.cache_dir))
            if not dataset_files(name, args.split)["qrels"].is_file():
                raise SystemExit(f"{name} has no {args.split} split; BEIR publishes dev only for "
                                 f"some datasets (nfcorpus, fiqa, quora-100k)")

    build_scorer(log_dir)
    clis: dict[str | None, Path] = {}
    # Build the default encoder last, so that ./warp-cli still points to it after the run.
    encoders = {c.encoder for c in configs if c.encoder is not None}
    for encoder in sorted(encoders, key=lambda e: e == MODERNBERT):
        clis[encoder] = build_cli(encoder, out_dir / "bin", log_dir)
    # Full-text runs need any CLI binary; use the first one that exists.
    clis[None] = next(iter(clis.values()), None) or build_cli(MODERNBERT, out_dir / "bin", log_dir)

    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else {}
    results.update({
        "started": results.get("started", dt.datetime.now().isoformat(timespec="seconds")),
        "git": git_state(), "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
        "split": args.split, "datasets": datasets, "configs": [asdict(c) for c in configs],
    })
    results.setdefault("builds", {})
    results.setdefault("runs", {})

    for name in datasets:
        files = dataset_files(name, args.split)
        slices = length_slices(files["corpus"], files["qrels"])
        for config in configs:
            build_dir = work_dir / name / config.build_key
            cost = build_index(name, config, clis[config.encoder], build_dir, args.rebuild)
            results["builds"].setdefault(name, {})[config.build_key] = asdict(cost)
            result = run_config(name, args.split, config, clis[config.encoder], build_dir, out_dir, slices)
            results["runs"].setdefault(name, {})[config.name] = asdict(result)
            results_path.write_text(json.dumps(results, indent=2))
        if not args.keep_indexes:
            for config in configs:
                shutil.rmtree(work_dir / name / config.build_key, ignore_errors=True)

    write_report(out_dir, results)
    print(f"report: {out_dir / 'report.md'}")
    return check_thresholds(results, args.check)


# --- reports -----------------------------------------------------------------

QUALITY = ["ndcg@10", "recall@100", "mrr@10", "success@1"]


def fmt(value, digits=3):
    return "–" if value is None else f"{value:.{digits}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def comparison_cell(comp: dict[str, float] | None) -> str:
    if comp is None:
        return "–"
    significant = comp["ci95_low"] > 0 or comp["ci95_high"] < 0
    mark = " **\\***" if significant else ""
    return f"{comp['delta']:+.3f} [{comp['ci95_low']:+.3f}, {comp['ci95_high']:+.3f}]{mark}"


def step_rows(out_dir: Path, results: dict, steps, metric: str) -> list[list[str]]:
    """One row for each step: the comparison on each dataset, then the gain / loss count."""
    runs, rows = results["runs"], []
    for base, cand, what in steps:
        row, gains, losses = [f"{base} → {cand}", what], 0, 0
        for d in results["datasets"]:
            comp = None
            if base in runs.get(d, {}) and cand in runs.get(d, {}):
                comp = compare_files(out_dir / runs[d][base]["per_query_file"],
                                     out_dir / runs[d][cand]["per_query_file"]).get(metric)
            if comp:
                gains += comp["ci95_low"] > 0
                losses += comp["ci95_high"] < 0
            row.append(comparison_cell(comp))
        rows.append(row + [f"{gains} / {losses}"])
    return rows


def refresh_slices(out_dir: Path, results: dict) -> None:
    """Calculate the slices again from the per-query files, for reports from older versions."""
    for d, configs in results["runs"].items():
        files = dataset_files(d, results["split"])
        lengths = length_slices(files["corpus"], files["qrels"])
        styles, overlaps = query_styles(d, results["split"]), query_overlaps(d, results["split"])
        for run_ in configs.values():
            rows = read_per_query(out_dir / run_["per_query_file"])
            run_["slices"] = slice_means(rows, lengths)
            run_["styles"] = slice_means(rows, styles)
            run_["overlap"] = slice_means(rows, overlaps)


def write_report(out_dir: Path, results: dict) -> None:
    configs = [c["name"] for c in results["configs"]]
    datasets = results["datasets"]
    runs, builds = results["runs"], results["builds"]
    lines = [
        "# Retrieval evaluation report", "",
        f"- Commit: `{results['git']['commit'][:12]}` (uncommitted changes: {results['git']['dirty']})",
        f"- Machine: {results['machine']}",
        f"- Split: `{results['split']}`. Use `dev` to choose settings. Use `test` only for the final report.",
        f"- Started: {results['started']}", "",
        "## Configurations", "",
        table(["name", "encoder", "keep", "query salience", "mode", "description"],
              [[c["name"], str(c["encoder"] or "–"), fmt(c["keep"], 2) if c["keep"] else "–",
                {None: "–", True: "on", False: "off"}[c["query_salience"]], c["mode"], c["description"]]
               for c in results["configs"]]), "",
        "## Datasets", "",
        table(["dataset", "docs", "queries", "content"],
              [[d, str(next(iter(builds.get(d, {}).values()), {}).get("docs", "–")),
                str(int(next(iter(runs.get(d, {}).values()), {}).get("metrics", {}).get("queries", 0))),
                dataset_note(d)] for d in datasets]), "",
    ]

    for metric in QUALITY:
        rows = []
        for d in datasets:
            rows.append([d] + [fmt(runs.get(d, {}).get(c, {}).get("metrics", {}).get(metric)) for c in configs])
        for label, agg in [("**mean**", lambda v: sum(v) / len(v)), ("**min**", min)]:
            cells = []
            for c in configs:
                values = [runs[d][c]["metrics"][metric] for d in datasets if c in runs.get(d, {})]
                cells.append(fmt(agg(values)) if len(values) == len(datasets) else "–")
            rows.append([label] + cells)
        lines += [f"## {metric}", "", table(["dataset"] + configs, rows), ""]

    steps = comparison_steps(configs) if len(configs) > 1 else []
    if steps:
        lines += ["## Ablation steps", "",
                  "Each row changes one thing, except the row \"pruning and query salience together\", "
                  "which shows the net change of the current default. A value is the paired mean difference "
                  "(candidate minus base) with a 95% bootstrap interval. "
                  "**\\*** marks an interval that does not include zero. "
                  "The last column counts the datasets with a significant gain and a significant loss.", ""]
        for metric in ["ndcg@10", "recall@100"]:
            rows = step_rows(out_dir, results, steps, metric)
            lines += [f"### {metric}", "", table(["step", "change"] + datasets + ["gain / loss"], rows), ""]

    lines += ["## Cost", "",
              "The index size is the size of the database and the index files on disk. "
              "The throughput counts the embed time and the index time. "
              "Configurations that differ only at query time share one index.", ""]
    rows = []
    for d in datasets:
        for c in results["configs"]:
            run_ = runs.get(d, {}).get(c["name"])
            build = builds.get(d, {}).get(Config(**c).build_key)
            if not run_ or not build:
                continue
            lat = run_["latency"]
            rows.append([d, c["name"], fmt(build["index_mb"], 1), fmt(build["docs_per_s"], 1),
                         fmt(build["embed_s"], 1), fmt(build["index_s"], 1), fmt(build["peak_rss_mb"], 0),
                         fmt(lat.get("p50_ms"), 0), fmt(lat.get("p95_ms"), 0)])
    lines += [table(["dataset", "config", "index MB", "docs/s", "embed s", "index s", "peak RSS MB",
                     "p50 ms", "p95 ms"], rows), ""]

    lines += ["## Document-length slices", "",
              "Each query goes into a bucket from the mean word count of its relevant documents. "
              "Pruning keeps a fraction of the tokens, so short documents have the highest risk.", ""]
    rows = []
    for d in datasets:
        for label, _, _ in LENGTH_BUCKETS:
            cells, count = [], None
            for c in configs:
                s = runs.get(d, {}).get(c, {}).get("slices", {}).get(label)
                cells.append(f"{s['ndcg@10']:.3f} / {s['recall@100']:.3f}" if s else "–")
                count = count or (s and int(s["queries"]))
            if count:
                rows.append([d, label, str(count)] + cells)
    lines += ["Each cell is NDCG@10 / Recall@100.", "",
              table(["dataset", "slice", "queries"] + configs, rows), ""]

    for key, title, order, note in [
        ("styles", "Query-style slices", None, "Each query style comes from make_known_item_set.py."),
        ("overlap", "Query-overlap slices", [b[0] for b in OVERLAP_BUCKETS],
         "The overlap is the fraction of the query's content words that also occur in the source "
         "document. BM25 is strong at a high overlap. Real user queries usually have a lower overlap."),
    ]:
        rows = []
        for d in datasets:
            found = {s for c in configs for s in runs.get(d, {}).get(c, {}).get(key, {})}
            labels = [label for label in order if label in found] if order else sorted(found)
            for label in labels:
                cells, count = [], None
                for c in configs:
                    s = runs.get(d, {}).get(c, {}).get(key, {}).get(label)
                    cells.append(f"{s['success@1']:.3f} / {s['mrr@10']:.3f}" if s else "–")
                    count = count or (s and int(s["queries"]))
                rows.append([d, label, str(count)] + cells)
        if rows:
            lines += [f"## {title}", "",
                      f"This section shows only known-item datasets. {note} "
                      "Each cell is Success@1 / MRR@10.", "",
                      table(["dataset", "slice", "queries"] + configs, rows), ""]

    lines += ["## Judged@10", "",
              "Judged@10 is the fraction of the returned top 10 that has a relevance judgment. "
              "A low value means that the qrels cannot measure this run well (pool bias). "
              "This problem is largest on trec-covid.", ""]
    rows = [[d] + [fmt(runs.get(d, {}).get(c, {}).get("metrics", {}).get("judged@10")) for c in configs]
            for d in datasets]
    lines += [table(["dataset"] + configs, rows), ""]

    (out_dir / "report.md").write_text("\n".join(lines))


def check_thresholds(results: dict, checks: list[str]) -> int:
    failed = 0
    for check in checks or []:
        match = re.fullmatch(r"([\w-]+):(?:([\w-]+):)?([\w@]+)>=([0-9.]+)", check)
        if not match:
            raise SystemExit(f"bad --check {check!r}; use DATASET[:CONFIG]:METRIC>=VALUE")
        dataset, config, metric, minimum = match.groups()
        config = config or results["configs"][0]["name"]
        value = results["runs"].get(dataset, {}).get(config, {}).get("metrics", {}).get(metric)
        ok = value is not None and value >= float(minimum)
        print(f"check {dataset}:{config}:{metric} = {fmt(value, 4)} >= {minimum}: {'ok' if ok else 'FAILED'}")
        failed += not ok
    return 1 if failed else 0


# --- the full pipeline ---------------------------------------------------------

PLANS = {
    "quick": {"suite": "smoke", "dev": ["nfcorpus"]},
    "standard": {"suite": "small", "dev": ["nfcorpus", "fiqa"]},
    "full": {"suite": "full", "dev": ["nfcorpus", "fiqa", "quora-100k"]},
}
STAGES = {
    "ablation": ("test", "ablation", "Ablation: each row adds or removes one change."),
    "sweep": ("dev", "sweep,min-sweep", "Sweep of the keep fraction and of the minimum, on dev splits."),
    "pruning-check": ("test", "pruning-check", "The default against the pruning candidates, on test splits."),
}


def known_item_sets() -> list[str]:
    return sorted(p.parent.name for p in TESTSET_DIR.glob("*/queries.jsonl"))


def pipeline(args) -> int:
    plan = PLANS[args.plan]
    known = known_item_sets() if args.known_item == "auto" else [
        k for k in args.known_item.split(",") if k and k != "none"]
    out_dir = Path(args.out or DEFAULT_OUT / f"pipeline-{dt.datetime.now():%Y%m%d-%H%M%S}").resolve()
    work_dir = Path(args.work_dir).resolve() if args.work_dir else out_dir / "indexes"
    suite = resolve_datasets(plan["suite"]) + known
    prepare([d for d in suite + plan["dev"] if d in DATASETS], Path(args.cache_dir))
    dev = [d for d in plan["dev"] + known if dataset_files(d, "dev")["qrels"].is_file()]
    print(f"pipeline: plan {args.plan}, test datasets {suite}, dev datasets {dev}")

    status = 0
    for stage in args.stages.split(","):
        split, configs, _ = STAGES[stage]
        datasets = suite if split == "test" else dev
        stage_args = argparse.Namespace(
            suite=",".join(datasets), configs=configs, keeps=args.keeps, mins=args.mins, split=split,
            out=str(out_dir / stage), work_dir=str(work_dir), cache_dir=args.cache_dir,
            rebuild=False, keep_indexes=True, check=args.check if stage == args.check_stage else None)
        print(f"pipeline: stage {stage} on {split} splits")
        status |= run(stage_args)
    write_summary(out_dir, args.stages.split(","))
    if not args.keep_indexes:
        shutil.rmtree(work_dir, ignore_errors=True)
    print(f"summary: {out_dir / 'summary.md'}")
    return status


def write_summary(out_dir: Path, stages: list[str]) -> None:
    lines = ["# Retrieval evaluation summary", ""]
    for stage in stages:
        path = out_dir / stage / "results.json"
        if not path.is_file():
            continue
        results = json.loads(path.read_text())
        configs = [c["name"] for c in results["configs"]]
        if not lines[2:]:
            lines += [f"- Commit: `{results['git']['commit'][:12]}` "
                      f"(uncommitted changes: {results['git']['dirty']})",
                      f"- Machine: {results['machine']}", ""]
        lines += [f"## {stage}", "", STAGES[stage][2],
                  f"Split: `{results['split']}`. Datasets: {', '.join(results['datasets'])}. "
                  f"Full report: [{stage}/report.md]({stage}/report.md).", ""]
        rows = []
        for metric in ["ndcg@10", "recall@100", "success@1"]:
            for label, agg in [("mean", lambda v: sum(v) / len(v)), ("min", min)]:
                cells = []
                for c in configs:
                    values = [results["runs"][d][c]["metrics"][metric]
                              for d in results["datasets"] if c in results["runs"].get(d, {})]
                    cells.append(fmt(agg(values)) if values else "–")
                rows.append([f"{metric} {label}"] + cells)
        lines += [table(["metric over datasets"] + configs, rows), ""]
        steps = comparison_steps(configs)
        if steps:
            step_table = step_rows(out_dir / stage, results, steps, "ndcg@10")
            lines += ["NDCG@10 steps. The gain / loss column counts the datasets with a significant "
                      "gain and a significant loss.", "",
                      table(["step", "change", "gain / loss"], [[r[0], r[1], r[-1]] for r in step_table]), ""]
    (out_dir / "summary.md").write_text("\n".join(lines))


def compare_dirs(args) -> int:
    base_dir, cand_dir = Path(args.base).resolve(), Path(args.candidate).resolve()
    base = json.loads((base_dir / "results.json").read_text())
    cand = json.loads((cand_dir / "results.json").read_text())
    regressions = 0
    lines = [f"# Comparison: `{base['git']['commit'][:12]}` → `{cand['git']['commit'][:12]}`", ""]
    for metric in ["ndcg@10", "recall@100", "mrr@10"]:
        rows = []
        for d, configs in cand["runs"].items():
            for c, run_ in configs.items():
                if c not in base["runs"].get(d, {}):
                    continue
                comp = compare_files(base_dir / base["runs"][d][c]["per_query_file"],
                                     cand_dir / run_["per_query_file"])[metric]
                if comp["ci95_high"] < 0 and -comp["delta"] >= args.tolerance:
                    regressions += 1
                rows.append([d, c, fmt(comp["base"]), fmt(comp["candidate"]), comparison_cell(comp),
                             fmt(comp["p_value"], 4)])
        lines += [f"## {metric}", "", table(["dataset", "config", "base", "candidate", "delta [95% CI]", "p"], rows), ""]
    lines.append(f"Significant regressions larger than {args.tolerance}: {regressions}")
    text = "\n".join(lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text)
    return 1 if regressions else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="download and convert datasets")
    p.add_argument("--suite", default="small", help=f"suite ({', '.join(SUITES)}) or dataset names")
    p.add_argument("--cache-dir", default=str(ROOT / ".cache" / "beir"))

    p = sub.add_parser("run", help="build, index, query and report")
    p.add_argument("--suite", default="smoke", help=f"suite ({', '.join(SUITES)}) or dataset names")
    p.add_argument("--configs", default="ablation",
                   help=f"comma list of configs ({', '.join(PRESETS)}) or sets ({', '.join(CONFIG_SETS)}, sweep)")
    p.add_argument("--keeps", default="0.2,0.3,0.5,1.0", help="keep fractions for the sweep set")
    p.add_argument("--mins", default="16,32,64", help="minimum kept tokens for the min-sweep set")
    p.add_argument("--split", default="test", choices=["test", "dev"])
    p.add_argument("--out", help="result directory (default: bench-results/eval/<time>)")
    p.add_argument("--work-dir", help="index directory (default: <out>/indexes)")
    p.add_argument("--cache-dir", default=str(ROOT / ".cache" / "beir"))
    p.add_argument("--rebuild", action="store_true", help="rebuild indexes that already exist")
    p.add_argument("--keep-indexes", action="store_true", help="do not delete indexes after each dataset")
    p.add_argument("--check", action="append", help="fail unless DATASET[:CONFIG]:METRIC>=VALUE")

    p = sub.add_parser("pipeline", help="run the ablation, the sweeps and the pruning check")
    p.add_argument("--plan", default="standard", choices=list(PLANS),
                   help="quick: NFCorpus; standard: 6 BEIR datasets; full: 10 BEIR datasets")
    p.add_argument("--stages", default="ablation,sweep,pruning-check",
                   help=f"comma list of stages ({', '.join(STAGES)})")
    p.add_argument("--known-item", default="auto",
                   help="known-item sets to add: auto (every set in testset/eval), none, or names")
    p.add_argument("--keeps", default="0.2,0.3,0.5,1.0")
    p.add_argument("--mins", default="16,32,64")
    p.add_argument("--out", help="result directory (default: bench-results/eval/pipeline-<time>)")
    p.add_argument("--work-dir", help="index directory, shared by all stages (default: <out>/indexes)")
    p.add_argument("--cache-dir", default=str(ROOT / ".cache" / "beir"))
    p.add_argument("--keep-indexes", action="store_true", help="keep the index directory at the end")
    p.add_argument("--check", action="append", help="fail unless DATASET[:CONFIG]:METRIC>=VALUE")
    p.add_argument("--check-stage", default="pruning-check", help="the stage that --check applies to")

    p = sub.add_parser("report", help="write report.md again")
    p.add_argument("out")

    p = sub.add_parser("compare", help="compare two result directories")
    p.add_argument("base")
    p.add_argument("candidate")
    p.add_argument("--tolerance", type=float, default=0.005,
                   help="ignore significant drops smaller than this (default: 0.005)")
    p.add_argument("--output", help="also write the Markdown comparison to this file")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(resolve_datasets(args.suite), Path(args.cache_dir))
        return 0
    if args.command == "run":
        return run(args)
    if args.command == "pipeline":
        return pipeline(args)
    if args.command == "report":
        out = Path(args.out).resolve()
        results = json.loads((out / "results.json").read_text())
        refresh_slices(out, results)
        (out / "results.json").write_text(json.dumps(results, indent=2))
        write_report(out, results)
        print(out / "report.md")
        return 0
    return compare_dirs(args)


if __name__ == "__main__":
    sys.exit(main())

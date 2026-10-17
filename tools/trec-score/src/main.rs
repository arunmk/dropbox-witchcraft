use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};

type Judgments = BTreeMap<String, BTreeMap<String, i64>>;
type DocumentMap = Option<BTreeMap<String, String>>;
type Run = BTreeMap<String, Vec<String>>;

const USAGE: &str = "\
Usage:
  trec-score RESULTS.tsv COLLECTION_MAP.json QRELS.json [CUTOFF]
      Print mean NDCG (default cutoff: 10).
  trec-score metrics RESULTS.tsv COLLECTION_MAP.json QRELS.json [--per-query OUT.tsv]
      Print every metric as METRIC<TAB>VALUE. Optionally write per-query values.
  trec-score compare BASE.tsv CANDIDATE.tsv [--samples N] [--seed S]
      Compare two per-query files from `metrics --per-query` with a paired
      bootstrap 95% interval and a paired randomization p-value.

Use a JSON null map for original document IDs.";

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("parse {}", path.display()))
}

fn read_run(reader: impl BufRead, remap: &DocumentMap) -> Result<Run> {
    let mut run = Run::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read result line {}", line_index + 1))?;
        let (query, documents) = line.split_once('\t')
            .filter(|(query, documents)| !query.is_empty() && !documents.contains('\t'))
            .with_context(|| format!("result line {}: expected query<TAB>doc,doc,...", line_index + 1))?;
        // score.py assigns -rank and overwrites duplicate documents and queries.
        let mut ranks = BTreeMap::new();
        for (rank, document) in documents.split(',').enumerate() {
            if document.is_empty() {
                continue;
            }
            let document = match remap {
                Some(map) => map.get(document).with_context(||
                    format!("result line {}: no document mapping for {document}", line_index + 1))?,
                None => document,
            };
            ranks.insert(document.to_owned(), rank);
        }
        let mut ranked: Vec<_> = ranks.into_iter().collect();
        ranked.sort_unstable_by_key(|(_, rank)| *rank);
        run.insert(query.to_owned(), ranked.into_iter().map(|(document, _)| document).collect());
    }
    Ok(run)
}

fn ndcg(documents: &[String], judgments: &BTreeMap<String, i64>, cutoff: usize) -> f64 {
    let dcg: f64 = documents.iter().take(cutoff).enumerate()
        .map(|(rank, document)| {
            let gain = judgments.get(document).copied().unwrap_or(0).max(0) as f64;
            gain / ((rank + 2) as f64).log2()
        }).sum();
    let mut ideal: Vec<_> = judgments.values().copied().filter(|gain| *gain > 0).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let ideal_dcg: f64 = ideal.iter().take(cutoff).enumerate()
        .map(|(rank, gain)| *gain as f64 / ((rank + 2) as f64).log2()).sum();
    if ideal_dcg == 0.0 { 0.0 } else { dcg / ideal_dcg }
}

fn is_relevant(judgments: &BTreeMap<String, i64>, document: &str) -> bool {
    judgments.get(document).is_some_and(|gain| *gain > 0)
}

fn recall(documents: &[String], judgments: &BTreeMap<String, i64>, cutoff: usize) -> f64 {
    let relevant = judgments.values().filter(|gain| **gain > 0).count();
    if relevant == 0 {
        return 0.0;
    }
    let found = documents.iter().take(cutoff).filter(|d| is_relevant(judgments, d)).count();
    found as f64 / relevant as f64
}

fn reciprocal_rank(documents: &[String], judgments: &BTreeMap<String, i64>, cutoff: usize) -> f64 {
    documents.iter().take(cutoff).position(|d| is_relevant(judgments, d))
        .map_or(0.0, |rank| 1.0 / (rank + 1) as f64)
}

fn success(documents: &[String], judgments: &BTreeMap<String, i64>, cutoff: usize) -> f64 {
    if documents.iter().take(cutoff).any(|d| is_relevant(judgments, d)) { 1.0 } else { 0.0 }
}

// The fraction of returned top-k documents that have any judgment. A low value
// means that the qrels cannot say much about this run (pool bias).
fn judged(documents: &[String], judgments: &BTreeMap<String, i64>, cutoff: usize) -> f64 {
    let top = documents.len().min(cutoff);
    if top == 0 {
        return 0.0;
    }
    documents.iter().take(cutoff).filter(|d| judgments.contains_key(*d)).count() as f64 / top as f64
}

type Metric = fn(&[String], &BTreeMap<String, i64>, usize) -> f64;

const METRICS: [(&str, Metric, usize); 9] = [
    ("ndcg@10", ndcg, 10),
    ("recall@10", recall, 10),
    ("recall@100", recall, 100),
    ("mrr@10", reciprocal_rank, 10),
    ("success@1", success, 1),
    ("success@5", success, 5),
    ("success@10", success, 10),
    ("judged@10", judged, 10),
    ("returned", |documents, _, _| documents.len() as f64, 0),
];

// Match pytrec_eval: use submitted queries with qrels, including empty runs.
fn judged_queries<'a>(run: &'a Run, judgments: &'a Judgments)
    -> impl Iterator<Item = (&'a String, &'a Vec<String>, &'a BTreeMap<String, i64>)> {
    run.iter().filter_map(|(query, documents)| {
        judgments.get(query).map(|qrels| (query, documents, qrels))
    })
}

fn evaluate(run: &Run, judgments: &Judgments, cutoff: usize) -> Result<f64> {
    if cutoff == 0 {
        bail!("cutoff must be positive");
    }
    let scores: Vec<_> = judged_queries(run, judgments)
        .map(|(_, documents, qrels)| ndcg(documents, qrels, cutoff)).collect();
    if scores.is_empty() {
        bail!("no result queries have relevance judgments");
    }
    Ok(scores.iter().sum::<f64>() / scores.len() as f64)
}

struct PerQuery {
    metrics: Vec<String>,
    rows: BTreeMap<String, Vec<f64>>,
}

fn per_query(run: &Run, judgments: &Judgments) -> Result<PerQuery> {
    let rows: BTreeMap<_, _> = judged_queries(run, judgments)
        .map(|(query, documents, qrels)| {
            let values = METRICS.iter().map(|(_, metric, k)| metric(documents, qrels, *k)).collect();
            (query.clone(), values)
        }).collect();
    if rows.is_empty() {
        bail!("no result queries have relevance judgments");
    }
    Ok(PerQuery { metrics: METRICS.iter().map(|(name, _, _)| name.to_string()).collect(), rows })
}

fn means(table: &PerQuery) -> Vec<f64> {
    let n = table.rows.len() as f64;
    (0..table.metrics.len())
        .map(|i| table.rows.values().map(|row| row[i]).sum::<f64>() / n).collect()
}

fn write_per_query(table: &PerQuery, out: &mut impl Write) -> Result<()> {
    writeln!(out, "query\t{}", table.metrics.join("\t"))?;
    for (query, values) in &table.rows {
        let values: Vec<_> = values.iter().map(|v| format!("{v}")).collect();
        writeln!(out, "{query}\t{}", values.join("\t"))?;
    }
    Ok(())
}

fn read_per_query(reader: impl BufRead) -> Result<PerQuery> {
    let mut lines = reader.lines();
    let header = lines.next().context("per-query file is empty")??;
    let mut columns = header.split('\t');
    if columns.next() != Some("query") {
        bail!("per-query header must start with 'query'");
    }
    let metrics: Vec<String> = columns.map(str::to_owned).collect();
    let mut rows = BTreeMap::new();
    for (line_index, line) in lines.enumerate() {
        let line = line?;
        let mut fields = line.split('\t');
        let query = fields.next().unwrap_or_default().to_owned();
        let values = fields.map(|v| v.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("per-query line {}: bad value", line_index + 2))?;
        if values.len() != metrics.len() {
            bail!("per-query line {}: expected {} values", line_index + 2, metrics.len());
        }
        rows.insert(query, values);
    }
    Ok(PerQuery { metrics, rows })
}

// SplitMix64: a small deterministic generator, so that reports are reproducible.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

struct Comparison {
    base: f64,
    candidate: f64,
    ci_low: f64,
    ci_high: f64,
    p_value: f64,
}

fn compare_paired(differences: &[f64], base: f64, samples: usize, rng: &mut Rng) -> Comparison {
    let n = differences.len();
    let delta = differences.iter().sum::<f64>() / n as f64;

    let mut boot: Vec<f64> = (0..samples).map(|_| {
        (0..n).map(|_| differences[rng.below(n)]).sum::<f64>() / n as f64
    }).collect();
    boot.sort_unstable_by(f64::total_cmp);
    let quantile = |q: f64| boot[((q * (samples - 1) as f64).round() as usize).min(samples - 1)];

    // Two-sided paired randomization test: flip the sign of each query's difference.
    let observed = delta.abs();
    let extreme = (0..samples).filter(|_| {
        let flipped: f64 = differences.iter()
            .map(|d| if rng.next_u64() & 1 == 0 { *d } else { -*d }).sum();
        (flipped / n as f64).abs() >= observed - 1e-12
    }).count();

    Comparison {
        base,
        candidate: base + delta,
        ci_low: quantile(0.025),
        ci_high: quantile(0.975),
        p_value: (extreme + 1) as f64 / (samples + 1) as f64,
    }
}

fn compare(base: &PerQuery, candidate: &PerQuery, samples: usize, seed: u64) -> Result<Vec<(String, Comparison)>> {
    if samples == 0 {
        bail!("samples must be positive");
    }
    let shared: Vec<&String> = base.rows.keys().filter(|q| candidate.rows.contains_key(*q)).collect();
    if shared.is_empty() {
        bail!("the per-query files share no queries");
    }
    if shared.len() != base.rows.len() || shared.len() != candidate.rows.len() {
        eprintln!("warning: comparing {} shared queries (base has {}, candidate has {})",
                  shared.len(), base.rows.len(), candidate.rows.len());
    }
    let mut rng = Rng(seed);
    let mut out = vec![];
    for (bi, metric) in base.metrics.iter().enumerate() {
        let Some(ci) = candidate.metrics.iter().position(|m| m == metric) else { continue };
        let differences: Vec<f64> = shared.iter()
            .map(|q| candidate.rows[*q][ci] - base.rows[*q][bi]).collect();
        let base_mean = shared.iter().map(|q| base.rows[*q][bi]).sum::<f64>() / shared.len() as f64;
        out.push((metric.clone(), compare_paired(&differences, base_mean, samples, &mut rng)));
    }
    Ok(out)
}

fn load_run(results: &Path, map: &Path, qrels: &Path) -> Result<(Run, Judgments)> {
    let remap: DocumentMap = read_json(map)?;
    let judgments: Judgments = read_json(qrels)?;
    let file = File::open(results).with_context(|| format!("open {}", results.display()))?;
    let run = read_run(BufReader::new(file), &remap)
        .with_context(|| format!("parse {}", results.display()))?;
    Ok((run, judgments))
}

fn option_value<'a>(args: &'a [String], name: &str) -> Result<Option<&'a str>> {
    match args.iter().position(|a| a == name) {
        Some(i) => args.get(i + 1).map(|v| Some(v.as_str())).with_context(|| format!("{name} needs a value")),
        None => Ok(None),
    }
}

fn positional(args: &[String], options: &[&str]) -> Vec<String> {
    let mut out = vec![];
    let mut i = 0;
    while i < args.len() {
        if options.contains(&args[i].as_str()) {
            i += 2;
        } else {
            out.push(args[i].clone());
            i += 1;
        }
    }
    out
}

fn run_metrics(args: &[String]) -> Result<()> {
    let files = positional(args, &["--per-query"]);
    if files.len() != 3 {
        bail!("{USAGE}");
    }
    let (run, judgments) = load_run(Path::new(&files[0]), Path::new(&files[1]), Path::new(&files[2]))?;
    let table = per_query(&run, &judgments)?;
    if let Some(path) = option_value(args, "--per-query")? {
        let file = File::create(path).with_context(|| format!("create {path}"))?;
        let mut out = BufWriter::new(file);
        write_per_query(&table, &mut out)?;
        out.flush()?;
    }
    println!("queries\t{}", table.rows.len());
    for (name, value) in table.metrics.iter().zip(means(&table)) {
        println!("{name}\t{value}");
    }
    Ok(())
}

fn run_compare(args: &[String]) -> Result<()> {
    let files = positional(args, &["--samples", "--seed"]);
    if files.len() != 2 {
        bail!("{USAGE}");
    }
    let samples = option_value(args, "--samples")?.map_or(Ok(10_000), str::parse)
        .context("--samples must be a positive integer")?;
    let seed = option_value(args, "--seed")?.map_or(Ok(42), str::parse)
        .context("--seed must be an integer")?;
    let read = |path: &str| -> Result<PerQuery> {
        let file = File::open(path).with_context(|| format!("open {path}"))?;
        read_per_query(BufReader::new(file)).with_context(|| format!("parse {path}"))
    };
    let rows = compare(&read(&files[0])?, &read(&files[1])?, samples, seed)?;
    println!("metric\tbase\tcandidate\tdelta\tci95_low\tci95_high\tp_value");
    for (metric, c) in rows {
        println!("{metric}\t{:.6}\t{:.6}\t{:+.6}\t{:+.6}\t{:+.6}\t{:.4}",
                 c.base, c.candidate, c.candidate - c.base, c.ci_low, c.ci_high, c.p_value);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--help" | "-h") => {
            println!("{USAGE}");
            Ok(())
        }
        Some("metrics") => run_metrics(&args[1..]),
        Some("compare") => run_compare(&args[1..]),
        _ => {
            if !(3..=4).contains(&args.len()) {
                bail!("{USAGE}");
            }
            let cutoff = match args.get(3) {
                Some(value) => value.parse::<usize>().context("cutoff must be a positive integer")?,
                None => 10,
            };
            let (run, judgments) = load_run(Path::new(&args[0]), Path::new(&args[1]), Path::new(&args[2]))?;
            println!("{}", evaluate(&run, &judgments, cutoff)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judgments(json: &str) -> Judgments {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn graded_gain_and_cutoffs_match_trec_eval() {
        let qrels = judgments(r#"{"q":{"a":2,"b":1,"unjudged":-1}}"#);
        let run = read_run(&b"q\tb,a,unjudged"[..], &None).unwrap();
        assert!((evaluate(&run, &qrels, 10).unwrap() - 0.8597186998521972).abs() < 1e-14);
        assert_eq!(evaluate(&run, &qrels, 1).unwrap(), 0.5);
        let ideal = read_run(&b"q\ta,b"[..], &None).unwrap();
        assert_eq!(evaluate(&ideal, &qrels, 10).unwrap(), 1.0);
    }

    #[test]
    fn only_submitted_judged_queries_enter_the_average() {
        let qrels = judgments(r#"{"q":{"a":1},"empty":{"a":1},"zero":{"a":0},"missing":{"a":1}}"#);
        let run = read_run(&b"q\ta\r\nempty\t\nzero\ta\nunknown\ta"[..], &None).unwrap();
        assert_eq!(evaluate(&run, &qrels, 10).unwrap(), 1.0 / 3.0);
        assert!(evaluate(&Run::new(), &qrels, 10).is_err());
        assert!(evaluate(&run, &qrels, 0).is_err());
    }

    #[test]
    fn remapping_and_duplicate_overwrites_preserve_ranking() {
        let remap = serde_json::from_str(r#"{"1":"a","2":"b","3":"a"}"#).unwrap();
        let run = read_run(&b"q\t1\nq\t1,,2,3"[..], &remap).unwrap();
        assert_eq!(run["q"], ["b", "a"]);
        assert!(read_run(&b"q\t4"[..], &remap).is_err());
    }

    #[test]
    fn malformed_results_are_reported() {
        for input in ["q", "\ta", "q\ta\tb"] {
            assert!(read_run(input.as_bytes(), &None).is_err());
        }
    }

    #[test]
    fn rank_metrics_use_positive_grades_only() {
        let qrels = &judgments(r#"{"q":{"a":0,"b":1,"c":2,"d":1}}"#)["q"];
        let docs: Vec<String> = ["x", "a", "b", "y", "c"].iter().map(|s| s.to_string()).collect();
        assert_eq!(recall(&docs, qrels, 10), 2.0 / 3.0);
        assert_eq!(recall(&docs, qrels, 2), 0.0);
        assert_eq!(reciprocal_rank(&docs, qrels, 10), 1.0 / 3.0);
        assert_eq!(reciprocal_rank(&docs, qrels, 2), 0.0);
        assert_eq!(success(&docs, qrels, 3), 1.0);
        assert_eq!(success(&docs, qrels, 2), 0.0);
        assert_eq!(judged(&docs, qrels, 10), 3.0 / 5.0);
        assert_eq!(judged(&[], qrels, 10), 0.0);
    }

    #[test]
    fn per_query_round_trip() {
        let qrels = judgments(r#"{"q1":{"a":1},"q2":{"b":1}}"#);
        let run = read_run(&b"q1\ta\nq2\tx,b"[..], &None).unwrap();
        let table = per_query(&run, &qrels).unwrap();
        let mut bytes = vec![];
        write_per_query(&table, &mut bytes).unwrap();
        let parsed = read_per_query(&bytes[..]).unwrap();
        assert_eq!(parsed.metrics, table.metrics);
        assert_eq!(parsed.rows, table.rows);
        let mrr = table.metrics.iter().position(|m| m == "mrr@10").unwrap();
        assert_eq!(means(&table)[mrr], 0.75);
    }

    #[test]
    fn paired_comparison_detects_consistent_gains_only() {
        let mut rng = Rng(7);
        let gains: Vec<f64> = (0..200).map(|i| 0.05 + 0.01 * ((i % 5) as f64 - 2.0)).collect();
        let c = compare_paired(&gains, 0.3, 2000, &mut rng);
        assert!((c.candidate - 0.35).abs() < 1e-9);
        assert!(c.ci_low > 0.0 && c.ci_high < 0.1);
        assert!(c.p_value < 0.01);

        let noise: Vec<f64> = (0..200).map(|i| if i % 2 == 0 { 0.1 } else { -0.1 }).collect();
        let c = compare_paired(&noise, 0.3, 2000, &mut rng);
        assert!(c.ci_low < 0.0 && c.ci_high > 0.0);
        assert!(c.p_value > 0.5);
    }
}

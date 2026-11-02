# Short-document regression cases

This directory holds 23 search queries whose relevant documents are short. With
the old pruning default (keep 30% of document tokens), Witchcraft did not return
any of these relevant documents in the top 100 results. Without pruning,
Witchcraft returned every relevant document in the top 100. The cases protect
against a return of this failure.

`cases.jsonl` is the case file. `scripts/short_doc_regression.py` selects the
cases and builds the datasets, and `make eval-regression` runs the check.

## The failure mode

At index time, Witchcraft keeps only the document tokens with the highest gate
scores. The number of kept tokens is a fraction of the document length. A long
document keeps many tokens, so some identifying tokens survive. A short document
keeps only a few tokens, and the gate can give its highest scores to generic
tokens.

Case `regress-short-cqa:54919` shows the failure. The query is
"rooting xperia tipo dual st21i2x". The only relevant document has 20 words and
30 tokens:

> How Can I Root my Sony Ericsson Xperia Tipo Dual? How can I root my Sony
> Ericsson Xperia Tipo Dual?

At keep 30%, the document keeps 9 tokens. `warp-cli saliency` shows the gate scores:

| Kept: the 9 highest gate scores | Removed |
| --- | --- |
| `?`, `Tip`, `root`, `Root`, `I`, `my` (twice), `X`, `per` | `Sony` (twice), `Eric`, `sson`, `ia`, `o`, `Dual` (twice), `How`, `can`, and the other tokens |

The kept tokens are mostly generic: "root", "I" and "my". The gate removed most
of the tokens that identify the phone: "Sony", "Ericsson", "Dual" and the "ia"
of "Xperia". 3,801 of the 22,998 documents in the corpus contain the word
"root", so after pruning, many other rooting questions match the query as well
as the relevant document does.

| Configuration | Rank of the relevant document |
| --- | --- |
| No pruning, query salience on | 2 |
| Keep 30%, query salience on (the default until 2026-10-17) | not in the top 100 |
| Keep 70%, query salience on (the current default) | 6 |

The other cases have the same pattern. The documents are short questions,
short emails (for example a read receipt, or a two-line note with a contact
name) and Wikipedia stubs (for example a village with one population number).

## Why these cases

1. **The cases come from measurements, not from guesses.** Each case is a real
   test query whose result changed from "found" to "not found" when pruning was
   on. The failure is in the stored ranks, so a reader can check each case.
2. **The cases test Recall@100, not only the order of the results.** NDCG@10
   can hide a document that drops from rank 40 to rank 150. A known-item user
   who looks for a short email or a short note does not find the file at all in
   that situation.
3. **Short documents are common in personal files.** Emails, notes, chat
   exports, short README files and stub pages are often shorter than 50 words.
   The public BEIR datasets have few short documents, so the mean scores of
   those datasets do not show this failure.
4. **The two sources cover two query types.**
   - `regress-short-cqa` (12 cases) uses BEIR CQADupStack (Android): short
     technical questions with human relevance labels. The corpus can be built
     again exactly, so the ranks can be compared between machines.
   - `regress-short-mix` (11 cases) uses the hard known-item set: paraphrased
     queries for short Enron emails and short Wikipedia pages. These queries use
     words that are different from the words of the document, which is the most
     difficult case for a pruned document.

## Selection criteria

`scripts/short_doc_regression.py select` chose the cases from the result
directory `bench-results/eval/keep70-check` (2026-10-17). A case must meet all
of these conditions:

1. The query is in the test split of CQADupStack (Android) or of the hard
   known-item set (`public-mix-hard`).
2. Every relevant document of the query has fewer than 50 words.
3. With no pruning and query salience on (`mb-full-qsal`), Recall@100 is 1.0:
   every relevant document is in the top 100.
4. With keep 30% and query salience on (`mb-default`), Recall@100 is 0.0: no
   relevant document is in the top 100.
5. No relevant document comes from GitHub. The licenses of GitHub files are
   mixed and unknown, and `cases.jsonl` stores the document text. This rule
   removed 1 case.

The other six eval datasets (NFCorpus, SciFact, ArguAna, SciDocs, FiQA and the
first known-item set) had no query that met conditions 2 to 4.

## The case file

Each line of `cases.jsonl` is one case:

| Field | Content |
| --- | --- |
| `case_id` | The regression dataset and the source query ID |
| `set` | `regress-short-cqa` or `regress-short-mix` |
| `source_dataset`, `source_query_id` | The eval dataset and the query ID in that dataset |
| `query`, `query_style` | The query text, and the style of a generated query (keyword, question or vague) |
| `relevant` | Each relevant document: ID, word count, source and license, and text |
| `ranks` | The rank of each relevant document in each configuration at selection time; `null` means not in the top 100 |

## How to run the check

```
make eval-regression
```

The target builds `regress-short-cqa` and `regress-short-mix` when the datasets
are missing, indexes both with the current default configuration, and checks
Recall@100:

| Dataset | Cases | Recall@100 at selection: no pruning / 30% / 70% | Check |
| --- | --- | --- | --- |
| `regress-short-cqa` | 12 | 1.00 / 0.00 / 0.92 | Recall@100 >= 0.83 |
| `regress-short-mix` | 11 | 1.00 / 0.00 / 0.82 | Recall@100 >= 0.72 |

Each check allows one more lost case than the current default. One case is
8% of the cases in `regress-short-cqa` and 9% of the cases in
`regress-short-mix`, and some documents are near rank 100 (for example ranks
89 and 97), so a small change to the ranking code can move one case across the
limit.

The distractors of each case are the full corpus of its source dataset, because
the failure depends on the competing documents. For `regress-short-cqa`, the
builder uses the BEIR corpus. For `regress-short-mix`, the builder uses
`datasets/eval/public-mix-hard.tsv` or `datasets/public-mix.tsv`. The GitHub
part of that corpus changes over time, so a corpus that is fetched again can
give slightly different ranks. The builder always adds the recorded text of each
case document.

## Manual tests

Two manual tests show the failure and the fix. Run both tests from the
repository root.

### Pass/fail test: semantic search (`make eval-regression`)

This test is deterministic, and it shows the failure clearly.

```
make eval-regression                                  # the current default: both checks pass
make eval-regression EVAL_REGRESSION_CONFIGS=mb-default   # the old 30% default: both checks fail
```

| Configuration | `regress-short-cqa` Recall@100 | `regress-short-mix` Recall@100 | Result |
| --- | --- | --- | --- |
| Keep 70% (current default) | 0.917 | 0.818 | pass |
| Keep 30% (`mb-default`) | 0.000 | 0.000 | fail |

The first run downloads CQADupStack (Android) and builds the indexes, which takes
approximately 10 minutes on an Apple Silicon Mac. The results are in
`bench-results/eval/regression-<configuration>/`.

### Statistical test: Pickbrain (`make pickbrain-short-doc-test`)

This test shows the effect in Pickbrain, the real application. In Pickbrain,
BM25 keyword search finds a short note when the query contains rare words of the
note, and Pickbrain shows only its best results. Also, the random seed of each
Pickbrain index moves single notes by many places. So this test reports a rate
over 420 queries, not a pass or a fail for single notes.

```
make pickbrain-short-doc-test                                   # 2 builds for each side
make pickbrain-short-doc-test PICKBRAIN_TEST_ARGS="--builds 3"  # more builds
env/bin/python scripts/pickbrain_short_doc_test.py --limit 20   # a quick check
```

The test writes the public mixed corpus as Claude memory notes into a temporary
directory, indexes the notes with keep 30% and with the default of `./pickbrain`,
and runs each query of `pickbrain-queries.jsonl` against both indexes. The test
uses a temporary `HOME` and `PICKBRAIN_DIR`, so the test never reads your real
sessions and never changes your real Pickbrain database. To compare with a
Pickbrain binary from master, use `--before-bin <path>`.

The test needs the public mixed corpus in `datasets/eval/public-mix-hard.tsv` or
`datasets/public-mix.tsv`. To make the corpus, run
`env/bin/python scripts/fetch_public_corpus.py datasets/public-mix.tsv --per-source 1000`.

Result on 2026-11-02, with 2 builds on an Apple Silicon Mac:

| Build | Notes shown, keep 30% | Notes shown, keep 70% | In the top 5, 30% / 70% | Shown only at 70% / only at 30% |
| --- | --- | --- | --- | --- |
| 1 | 284 of 420 (67.6%) | 306 of 420 (72.9%) | 239 / 271 | 38 / 16 |
| 2 | 282 of 420 (67.1%) | 300 of 420 (71.4%) | 239 / 255 | 33 / 15 |

Keep 70% showed 18-22 more short notes than keep 30% in both builds. Single notes
move in both directions, so look at the rates, not at single notes.

`pickbrain-queries.jsonl` holds the 420 known-item queries whose document has
fewer than 50 words (130 Enron emails and Wikipedia pages), with the text of
each document. `scripts/short_doc_regression.py select-pickbrain` writes the file.

## Limits

- **The cases are a regression check, not a quality measure.** The cases were
  selected because the 30% configuration failed on them, so the scores are
  biased by design. Do not report the Recall@100 of these sets as search
  quality, and do not tune settings on these sets.
- **The set is small.** 23 cases find a return of the failure, but the cases
  cannot measure a small change in quality.
- **The current default does not pass every case.** At keep 70%, 3 of the 23
  relevant documents are still outside the top 100. Those cases show the
  remaining risk for short documents.

## Sources and licenses

- CQADupStack (Android) comes from BEIR and from Stack Exchange; Stack Exchange
  content is licensed CC BY-SA.
- The Enron emails come from the public FERC release, through the Hugging Face
  dataset `corbt/enron-emails`.
- The Wikipedia pages come from `wikimedia/wikipedia` (20231101.en); Wikipedia
  text is licensed CC BY-SA.
- Claude wrote the queries of `regress-short-mix` with
  `scripts/make_known_item_set.py`.

## How to add or refresh cases

Run an eval that contains the reference configuration and the pruned
configuration on the source datasets, then select again:

```
scripts/eval_suite.py run --suite cqadupstack-android,public-mix-hard \
    --configs mb-full-qsal,mb-default,mb-keep070-qsal --out bench-results/eval/keep70-check
scripts/short_doc_regression.py select --results bench-results/eval/keep70-check
scripts/short_doc_regression.py build
```

Review the changes to `cases.jsonl` before a commit, and update the checks in
the `Makefile` when the number of cases or the default configuration changes.

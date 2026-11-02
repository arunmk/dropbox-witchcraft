# Regression test sets

This folder holds regression test sets for Witchcraft search. A regression test
set contains real queries that the search once failed on, with the correct
documents for those queries. A regression check runs the queries again and
fails if the search loses those documents again.

## Why the project has regression test sets

The benchmark datasets (NFCorpus, FiQA, the known-item sets and the others in
`scripts/eval_suite.py`) measure the mean quality over hundreds or thousands of
queries. A mean can hide a specific failure. For example, a change can lose 12
short documents completely and still change the mean NDCG@10 of a dataset by
less than 0.01, which is inside the noise of the benchmark.

A regression test set makes a known failure visible again. Each set holds only
the queries of one failure, so a return of that failure changes the result of
the set by a large amount, and the check fails.

## Why this folder has the short-document set

In October 2026, the evaluation of document token pruning found that the old
default (keep 30% of document tokens) removed short documents from the search
results. A short document keeps only a few tokens after pruning. The gate often
keeps generic tokens, for example "root", "I" and "my", and removes the tokens
that identify the document, for example the name of a phone. Thousands of other
documents contain the same generic tokens, so the short document falls below
rank 100, and a user who looks for that document does not find it.

These commits caused the failure and fixed the failure:

| Commit | Date | Change |
| --- | --- | --- |
| `e894ab1` | 2026-06-25 | "Add learned token gate pruning": added gate pruning as an option. Pruning ran only when `WARP_DOC_TOKEN_KEEP_FRACTION` was set. |
| `ae22687` | 2026-07-03 | "Bake in salience token pruning defaults": made pruning the default, with a keep fraction of 0.25. From this commit on, every document was pruned. |
| `b22c6a2` | 2026-07-29 | "default to keeping 30% of tokens": changed the default keep fraction from 0.25 to 0.3. |
| `e73fd7d` | 2026-07-31 | "Fix tests for ModernBERT with token pruning": changed the unit tests to longer documents and lower score thresholds, because "short documents lose too much signal". The commit changed the tests, not the pruning, so the failure stayed in the search. |
| this branch | 2026-10-17 | "Add a retrieval evaluation suite and keep 70% of document tokens by default" (branch `retrieval-eval-keep70`): measured the failure and changed the default keep fraction from 0.3 to 0.7. |

The failure was important for two reasons:

1. Personal files contain many short documents: emails, notes, short README
   files and stub pages.
2. The mean scores of the benchmark datasets did not show the failure clearly,
   because most benchmark documents are long.

The project changed the default keep fraction from 30% to 70% on 2026-10-17. The
short-document set makes sure that a later change to pruning, to the gate model
or to the ranking code does not bring the failure back without a failed check.

| Set | Folder | Cases | What the set protects |
| --- | --- | --- | --- |
| Short documents | [`short-docs/`](short-docs/README.md) | 23 | Short questions, emails and Wikipedia stubs that pruning once removed from the top 100 |

`short-docs/README.md` explains the failure in detail, with an example document,
the tokens that pruning kept and removed, and the ranks before and after the
change.

## How a regression set differs from a benchmark dataset

| Property | Benchmark dataset | Regression set |
| --- | --- | --- |
| Query selection | All the queries of the dataset | Only queries that failed |
| What the score means | The search quality on that type of content | Whether a known failure has come back |
| Use for tuning | Yes, on the dev split | No: the queries were selected because of a failure, so the score is biased |
| Size | Hundreds or thousands of queries | Tens of queries |
| Check | A minimum value with a margin for noise | A minimum value that allows a small number of extra lost cases |

Do not report the score of a regression set as search quality.

## When to run the check

Run `make eval-regression` before you merge a change to one of these parts:

- document token pruning, for example `WARP_DOC_TOKEN_KEEP_FRACTION` or `WARP_DOC_TOKEN_MIN_KEEP`;
- the gate model or the encoder weights;
- query salience;
- the scoring, the index clustering or the ranking code.

The check embeds approximately 26,000 documents, so the CI workflow does not run
the check on each pull request. Run the check on a machine with a GPU, for
example an Apple Silicon Mac.

## How to add a regression set

1. Find the failure in the saved results of an eval run (`bench-results/eval/`).
2. Write the selection rule as code, so that another person can select the
   same cases again. `scripts/short_doc_regression.py` is an example.
3. Put the cases and a README in a new subfolder. The README must explain the
   failure, the selection criteria, the check and the limits of the set.
4. Add the datasets to `scripts/eval_suite.py`, and add the checks to the
   `Makefile`.
5. Add a row for the set to the table in this file.

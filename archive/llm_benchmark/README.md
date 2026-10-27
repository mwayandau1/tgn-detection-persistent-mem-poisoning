# Archive: the withdrawn LLM-generated benchmark

**Not part of the final pipeline.** Nothing in `src/` imports these files, and no number in
`experiments/audit/RESULTS_FREEZE.md` comes from them. They are kept because they are the
evidence for one finding the thesis and paper can report: a corpus whose poisoned records are
*generated* by a language model carries its label in the text.

## What happened

The study's first benchmark (`data/raw/llm_generated`) generated each record with a language
model, poisoned records from "harmful advice" prompts and benign ones from topic prompts. A
logistic regression given a single record's 384-dimensional embedding, with no graph and no
temporal order, scored AUROC 1.000 on it. The label was recoverable from one record's text, so
no graph or temporal model was being tested. That benchmark was withdrawn and replaced by
corpora built from unmodified knowledge-base entries (`src/data_generation/agentpoison_sequences.py`),
on which the same probe scores 0.5300 (ap_burst2k) and 0.5143 (ap_burst2k_nodup), against a
base rate of 0.0256.

## Files

| File | What it does |
|---|---|
| `llm_generator.py` | Generated the withdrawn corpus (needs a paid LLM API key) |
| `rebuild_session_structure.py` | An attempt to repair that corpus by giving benign records session structure; it did not remove the leak |
| `leak_mechanism.py` | Measures why the leak exists: probe AUROC as the training set is thinned, as test prevalence falls, and as embedding dimensions are removed; compares against ap_burst2k |
| `default_llm_era.yaml` | The configuration these scripts read (attack types, domain and API settings that the final pipeline no longer uses) |

The corpus itself, `data/raw/llm_generated` (715 MB), stays on disk outside Git.

## Running them

From the repository root, for example:

```
python -m archive.llm_benchmark.leak_mechanism --data data/raw/llm_generated --contrast data/raw/ap_burst2k
```

The leak measurements reported earlier are in Git history (tag `pre-cleanup`,
`experiments/reported_results.json`, key `leak_diagnosis`).

# Step 6 plan (updated Tue 03 Nov 09:48)

Benchmark: `data/raw/ap_burst2k_nodup`; configuration: step 5 (see `experiments/IMPROVEMENT.md`). Estimated finish: **Tue 03 Nov, 09:48** (about 0 h left).

| Phase | What | Estimate | State |
|---|---|---|---|
| A | Detector and static GAT to five seeds | 21.0 h | done |
| B | Online scores and R2 rebuild (records, probes, figures, sweeps) | 18.0 h | done |
| D1 | Headline summaries (detector, static GAT), each run twice | 0.5 h | done |
| C | Ablations x 5 seeds (memory only, aggregator only, widened static, mean pool, no structural features, no coherence features) | 80.0 h | done |
| D2 | All summaries twice, oracle comparison, profile, duplicates cross-score | 3.0 h | done |

Detail: training `experiments/seeds_coherence/STATUS.md`; rebuild `experiments/rebuild_coherence/STATUS.md`.

All phases finished.

- experiments\seeds_coherence\summary_headline.json: identical on two runs
- experiments\seeds_coherence\seed_tables.json: identical on two runs
- experiments\seeds_coherence\ensemble_headline.json: identical on two runs
- experiments\seeds_coherence\summary.json: identical on two runs
- experiments\seeds_coherence\oracle_comparison_5seeds.json: identical on two runs
- experiments\seeds_coherence\cross_duplicates.json: identical on two runs

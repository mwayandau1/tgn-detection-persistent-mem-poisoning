# Main-benchmark plan (updated Fri 23 Oct 09:50)

Benchmark: `data/raw/ap_burst2k_nodup`. Estimated finish: **Fri 23 Oct, 09:50** (about 0 h left).

| Phase | What | Estimate | State |
|---|---|---|---|
| A | Detector and static GAT to five seeds | 7.5 h | done |
| B | R2 rebuild (records, probes, figures, sweeps) | 16.0 h | done |
| C | Ablations x 5 seeds (memory only, aggregator only, size-matched static, mean pool, no structural features) | 55.5 h | done |
| D | Latency baseline and summaries, each run twice | 1.0 h | done |

Detail: training `experiments/seeds_nodup/STATUS.md`; rebuild `experiments/rebuild/STATUS.md`.

All phases finished.

- experiments\seeds_nodup\summary.json: identical on two runs
- experiments\seeds_nodup\seed_tables.json: identical on two runs
- experiments\seeds_nodup\oracle_comparison_5seeds.json: identical on two runs

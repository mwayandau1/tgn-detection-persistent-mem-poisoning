# Commit hashes before and after the history rewrite of 27 October 2026

On 27 October 2026 the commit messages on `benchmark-hardening` were edited to remove
co-author lines. Only the messages changed: every commit keeps its tree (the files), author,
date and subject, which was checked commit by commit. Because a commit's hash covers its
message, every hash changed. Run records and documents written before the rewrite cite the
old hashes (for example `"git_commit": "600f647"` in each `run_config.json`); this table
gives the commit each one now corresponds to, with identical code.

| Old | New | Subject |
|---|---|---|
| `a6a80fc` | `876377f` | Fix benchmark label leaks, implement semantic_coherence, add difficulty control |
| `2ea5c24` | `666985c` | Add REPRODUCE.md, benchmark difficulty preset, and gap-0.125 run artifacts |
| `45ddede` | `bdf8b67` | Make the LLM generator testable offline; verify keys up front; fix injection count |
| `1c29d74` | `a1ee85f` | Fix the benign baseline in the separability report; set a live OpenRouter default |
| `0345c4a` | `c2f4387` | Match cluster size when computing the coherence gap |
| `a22a29f` | `867beb8` | Vary attack size per sequence; report detection by attack type and size |
| `e6468f1` | `c4b786b` | Add split-ratio overrides to the LLM generator |
| `64843c7` | `6c22599` | Add a node-level content control; it invalidates the LLM benchmark |
| `97db539` | `ff371ed` | Build sequences from AgentPoison's real KB; add a scale-aware readout |
| `4e4d0b2` | `32f3826` | Fix the scale-aware readout: bounded mean, not an unbounded sum |
| `a90eb80` | `bd78e6e` | Build a benchmark that actually tests temporal modelling, and a gate that proves it |
| `6983e7a` | `0050c5a` | Add a gated scale readout: +771 parameters, identical to the baseline at init |
| `1497974` | `784df4a` | Evaluate at sequence level, at a calibrated FPR, and against the right baseline |
| `39553bc` | `fd5749f` | Record the measured results |
| `bddb643` | `b30b601` | Make the notebook a driver over src/ instead of a fork of it |
| `a8ae005` | `3d3200a` | Record the calibrated detection-latency result |
| `387d952` | `c6596f3` | Measure why the content leak exists and whether rebalancing removes it |
| `bf40b11` | `6f64736` | Re-run the thesis's own baselines on the replacement benchmark |
| `7d01a90` | `17ea348` | Make the notebook produce the figures the thesis reports |
| `4475257` | `485c174` | Add a reproduction notebook for the ap_burst2k results |
| `48dcb86` | `1d69839` | Add ap_burst2k baseline and ablation notebooks; mark the stale ones superseded |
| `dbcef9d` | `507f1e2` | Restore two walkthrough documents deleted by mistake, and banner them |
| `86587fb` | `8158e58` | Complete the ablation study: mean-pool readout and structural features |
| `513c69d` | `aff86e9` | Stop the notebooks hardcoding their reference values |
| `fa97719` | `e5a1ae9` | Remove the superseded benchmarks, runs and generator |
| `fa40718` | `f88ef08` | Regenerate the Chapter 4 figures on ap_burst2k |
| `600f647` | `c1652cc` | Re-measure the robustness sweeps on ap_burst2k, and fix four defects found doing it |
| `1ed7072` | `e7324fa` | Freeze the results on a duplicate-free benchmark, with five seeds per model |

# Thesis change list (brief v2, Section 2, plus audit findings)

For: Moses, applying edits to `MosesWinboodAyandauTGNN-3.docx` (13 Oct 2026).
Evidence for every factual change is in `AUDIT.md` in this folder; section numbers there are
given as "Audit 1.2" and so on.

Conventions: each item gives location, current text, proposed text and reason. Proposed text
keeps the surrounding sentence and changes as little as possible. British English, no em or en
dashes in prose, no equations.

**Updated 23 October 2026:** group C now follows `RESULTS_FREEZE.md` (main corpus
ap_burst2k_nodup). Groups A and B are unchanged except A13 and A21b, which point to C.

**Updated 3 November 2026:** group D (at the end) follows `RESULTS_FREEZE_2.md`, the improved
configuration, under the framing Moses chose on 3 November: the contribution is the
arrival-order representation read by a graph model over the whole trajectory, and the thesis
states that the temporal components add nothing measurable on this corpus. Where D and C touch
the same passage or table, D replaces C; C items D does not mention still apply.

Three groups:

* **A. Apply now.** Facts that no further experiment will change.
* **B. Apply now with interim numbers.** Wording is settled; numbers are single-seed and will
  be replaced by the multi-seed figures from E2.
* **C. Wait for E2 and E3.** Listed so nothing is forgotten; text to be supplied with results.

---

## A. Apply now

### A1. Section 1.4, lead-in (grammar)
* Current: "The objectives of the study are as follows to."
* Proposed: "The objectives of the study are to:"
* Reason: brief item 5; the numbered objectives begin with bare verbs.

### A2. Objective I, Objective IV, Research Question IV, Section 1.9 (attack strategy)
* Objective I. Current: "across representative attack strategies." Proposed: "under a
  flood-placement threat model."
* Objective IV. Current: "the sensitivity of detection performance to attack type and to
  degradation of the input data". Proposed: "the sensitivity of detection performance to
  attack size and to degradation of the input data".
* Research Question IV. Current: "how does performance vary under attack type and input
  degradation". Proposed: "how does performance vary with attack size and under input
  degradation".
* Section 1.9, second paragraph. Current: "the performance under each attack strategy".
  Proposed: "performance as a function of attack size".
* Reason: brief item 5; one threat model is simulated and only attack size is varied.

### A3. Section 1.6, second paragraph (attacks "have all been published")
* Current: "The attacks are not novel as they have all been published in the peer-reviewed
  or preprint literature."
* Proposed: "The attack is not novel: placing several records in one region of the embedding
  space is a simplified form of the multi-record injection described by Chen et al. (2024)
  and Zou et al. (2025), and the simulator implements no technique beyond theirs."
* Reason: brief item 11; the flood-placement variant is the study's own construction.

### A4. Section 1.6, third paragraph (MemGPT claim)
* Current: "having been shown against production systems such as MemGPT and similar
  retrieval-enhanced models (Packer et al., 2023; Dong et al., 2025)."
* Proposed: "having been demonstrated against a web-shopping agent, a clinical records agent
  and a question-answering agent (Dong et al., 2025), all of which retrieve memory by the
  same similarity mechanism as MemGPT (Packer et al., 2023)."
* Reason: brief item 11. MINJA is evaluated on RAP (Webshop), EHRAgent (MIMIC-III and eICU)
  and a QA agent on MMLU; MemGPT is not evaluated (checked against the full paper,
  arXiv 2503.03704v5).

### A5. Section 2.1 (section count)
* Current: "organised into seven substantive sections"
* Proposed: "organised into eight substantive sections"
* Reason: brief item 9; Sections 2.2 to 2.9.

### A6. Sections 1.2 and 2.4 (Xiang et al., 2024 is BadChain, an attack)
* Section 1.2. Current: "(Mu et al., 2026; Xiang et al., 2024)". Proposed: "(Mu et al., 2026)".
* Section 2.4. Current: "exemplified by the behavioural approach of Mu et al. (2026) and the
  supervisory framework of Xiang et al. (2024), compares current output". Proposed:
  "exemplified by the behavioural approach of Mu et al. (2026), compares current output".
* Keep the reference: Section 2.3 cites BadChain correctly ("backdoors can be installed
  through poisoned chain-of-thought demonstrations").
* Optional replacement, only if a second example is wanted in 2.4: GuardAgent (Xiang et al.,
  2025, ICML; arXiv 2406.09187). It must be described as what it is, a guard agent that
  checks each proposed action against stated safety requests by generating and executing
  guardrail code, because it does not compare output against a baseline distribution.
* Reason: brief item 16; verified on the arXiv abstract pages of both papers.

### A7. Section 2.8, last paragraph (Shafran et al., 2025)
* Current: "Shafran et al. (2025) treat the retrieval corpus as a static knowledge base into
  which anomalous individual records have been inserted, and pose the security problem as the
  detection of those records."
* Proposed: "Shafran et al. (2025) show that a single inserted blocker document can deny a
  retrieval-augmented system's answer to a targeted query, an attack conceived against a
  static corpus and carried by individual inserted records."
* Reason: brief item 17; the paper is a denial-of-service ("jamming") attack, not a detection
  study. The next sentence ("The proposed framework extends that conception...") still reads.

### A8. Section 3.1 (three threat models; 3.15 vs 3.16)
* Current: "The attack simulation and the three threat models are defined in Section 3.6."
  Proposed: "The attack simulation and the threat model are defined in Section 3.6."
* Current: "reliability, validity and reproducibility in Section 3.14 and the chapter is
  summarised in Section 3.15." Proposed: "reliability, validity and reproducibility in
  Section 3.14, the benchmark validation protocol in Section 3.15, and the chapter is
  summarised in Section 3.16."
* Reason: brief item 1.

### A9. Section 3.5, Table 3.1, Section 3.8, Table 3.6 (387 vs 391; structural features)
* Section 3.5, second paragraph. Current: "a feature vector of 387 dimensions". Proposed: "a
  feature vector of 391 dimensions".
* Section 3.5, third paragraph, add after the retrieval recency definition: "Four further
  dimensions describe the record's position in the graph at that checkpoint: its number of
  incident edges divided by 50, the same count divided by the number of other records in the
  snapshot, and the mean and the maximum similarity carried by its incident edges. They are
  recomputed at every snapshot from that snapshot's edges."
* Table 3.1, add four rows:
  "387 | 1 | Degree (fixed scale) | Number of incident edges divided by 50";
  "388 | 1 | Degree (relative) | Number of incident edges divided by the number of other records";
  "389 | 1 | Mean incident weight | Mean cosine similarity over the record's edges";
  "390 | 1 | Maximum incident weight | Largest cosine similarity over the record's edges".
* Table 3.6. Current: "387 (384 semantic, 3 behavioural)". Proposed: "391 (384 semantic,
  3 behavioural, 4 structural)".
* Section 3.8, first paragraph, "described in Section 3.5" then becomes true; no change.
* Reason: brief item 2; `dataset.py` `_structural_features`.

### A10. Section 3.5 and Table 3.1 (retrieval frequency normalisation)
* Section 3.5. Current: "normalised by the maximum retrieval count within the same
  trajectory". Proposed: "divided by a fixed constant of 50".
* Same paragraph. Current: "All three were normalised within the trajectory to which the
  record belonged, so that no normalisation statistic was computed across trajectories or
  across partitions". Proposed: "All three were scaled by quantities internal to the
  trajectory or by fixed constants, so that no normalisation statistic was computed across
  trajectories or across partitions".
* Table 3.1, row 385. Current: "Retrieval count normalised by the trajectory maximum".
  Proposed: "Retrieval count divided by 50".
* Reason: Audit 1.5; `agentpoison_sequences.py` line 238.

### A11. Section 3.5 and Table 3.2 (edge rule)
* Section 3.5. Current: "met or exceeded a threshold of 0.6". Proposed: "exceeded a threshold
  of 0.6".
* Table 3.2. Current: "Edge created where similarity is greater than or equal to 0.6".
  Proposed: "Edge created where similarity is greater than 0.6".
* Reason: Audit 1.5; the generator tests `sim > threshold`.

### A12. Section 3.6, Table 3.3, Section 4.5, Table 4.3 (cluster size)
* Section 3.6. Current: "varies between 5 and 24 across the corpus". Proposed: "varies between
  5 and 20 across the corpus".
* Table 3.3. Current: "5 to 24 records". Proposed: "5 to 20 records".
* Section 4.5. Current: "The corpus varies attack size between 5 and 24 records". Proposed:
  "The corpus varies attack size between 5 and 20 records".
* Table 4.3, first column. Current: "1 to 8" and "17 to 24". Proposed: "5 to 8" and "17 to 20".
* Reason: brief items 7 and E4; `metadata.json` holds 5 to 20, the generator default; the
  table labels were bin edges hard-coded in `generate_eval_results.py`.

### A13. Section 3.6 and Table 3.3 (arrival pattern and duplication)
* **Updated 23 Oct:** the added sentence on repeated entries below is superseded by C0, since the
  main corpus no longer repeats entries. The arrival-pattern changes still apply.
* Section 3.6, fourth paragraph. Current: "drawn from an equally narrow neighbourhood but
  arriving gradually across 120 interactions". Proposed: "drawn from an equally narrow
  neighbourhood but arriving at random positions within a span of up to 120 interactions,
  so that its first and last records lie 75 interactions apart at the median, against
  consecutive interactions for the attack".
* Add after "The rate at which the cluster accumulates is therefore the only quantity
  separating the two classes, which is the property the detector is required to recover.":
  "Both clusters are drawn from the eight nearest neighbours of a randomly chosen entry, so a
  cluster of more than eight records necessarily repeats entries, in the attack and in its
  control alike."
* Table 3.3, arrival pattern. Poisoned: "One injected record per consecutive interaction".
  Clean: "Random positions within up to 120 interactions (median span 75)".
* Reason: Audit 1.4 and 1.5; the duplication rule is what Section 4.5's size effect tracks,
  so the method has to disclose it.

### A14. Section 3.7 (domain; stratification; prevalence)
* Current: "within a customer support domain". Proposed: "drawn from a general-knowledge
  question-answering domain".
* Current: "The allocation was stratified by class, so that poisoned sequences constitute
  40.0% of each of the three partitions and of the dataset as a whole." Proposed: "The
  allocation was a seeded random shuffle and was not stratified by class; it placed exactly
  40.0% poisoned sequences in each of the three partitions, as in the dataset as a whole."
* Leave "approximately 2.6%" (the test figure is 2.54% to 2.56% depending on the sample).
* Section 3.16. Current: "its stratified partitioning by sequence". Proposed: "its
  partitioning by sequence".
* Section 6.2. Current: "partitioned strictly and with stratification by sequence". Proposed:
  "partitioned strictly by sequence".
* Section 6.3. Current: "which allocates by sequence with stratification". Proposed: "which
  allocates by sequence".
* Reason: brief item 10; Audit 1.5.

### A15. Section 3.9 and Table 3.6 (training record)
* First and second paragraphs ("Training proceeded in two sequential phases..." and "In the
  first phase..."): replace both with:
  "A supervised contrastive pre-training phase (Khosla et al., 2020) was implemented during
  development but was not run for the reported detector or for any variant compared with it,
  so every model in Chapter Four was trained from random initialisation under the objective
  described below."
* Third paragraph. Current: "In the second phase the full network was fine-tuned end to end".
  Proposed: "The full network was trained end to end".
* Same paragraph. Current: "class weighting 0.92". Proposed: "class weighting 0.85".
* Same paragraph. Current: "where fewer than 2% of records are injected". Proposed: "where
  approximately 2.5% of records are injected".
* Same paragraph. Current: "weights of 1.0 on the graph-level term and 2.5 on the node-level
  term". Proposed: "weights of 1.0 on the graph-level term and 1.5 on the node-level term".
* Same paragraph. Delete: "The contrastive objective was not retained during the second phase."
* Fourth paragraph. Current: "applied during second-phase training". Proposed: "applied during
  training".
* Fifth paragraph. Current: "The second phase was configured for 10 epochs with early stopping
  on validation loss at a patience of 10 epochs, and the checkpoint attaining the lowest
  validation loss was retained for evaluation. The patience was set at the epoch budget
  deliberately, so that the configured schedule would run to completion, with early stopping
  retained as a safeguard against divergence, with no active constraint on training length."
  Proposed: "Training ran for 12 epochs with early stopping at a patience of 10 epochs, which
  did not trigger, and the checkpoint attaining the highest validation area under the
  receiver operating characteristic curve at window level was retained for evaluation.
  Selection on that metric rather than on validation loss was fixed before training, because
  the two diverge once a model begins to overfit and selection on loss can retain a
  checkpoint that is worse on the metric being reported."
* Table 3.6. Delete "Contrastive temperature | 0.07". Change: "Node-level objective | Focal
  loss, focusing 3.0, class weighting 0.85"; "Objective weights (graph, node) | 1.0, 1.5";
  "Epochs (phase one, phase two) | 5, 10, early stopping patience 10" to "Epochs | 12, early
  stopping patience 10 (not triggered)"; add "Checkpoint selection | Highest validation
  window-level AUROC".
* Section 3.16. Current: "documented the two-phase training procedure". Proposed: "documented
  the training procedure".
* Section 4.2. Current: "Training followed the two-phase procedure specified in Section 3.9.
  The second phase converged steadily and without instability." Proposed: "Training followed
  the procedure specified in Section 3.9 and converged steadily and without instability."
* Section 6.2. Current: "the two-phase training procedure". Proposed: "the training procedure".
* Reason: brief E7; Audit 2. The training log has 12 epochs and selected epoch 9 by AUROC;
  no run directory contains a phase-one checkpoint; the recorded command passes
  `--skip-phase1 --epochs-supervised 12` and no loss overrides, so `configs/default.yaml`
  (0.85, 1.5) applied. If Khosla et al. (2020) is dropped as well, remove it from the
  reference list.

### A16. Section 3.10 (two nine-feature regressions)
* Current: "The window-level benchmark fitted a logistic regression to the nine features
  specified in Section 3.11, computed on the final snapshot of each window."
* Proposed: "The window-level benchmark fitted a logistic regression to the nine features
  specified in Section 3.11, computed on the final snapshot of each window, standardised, and
  with the tightest neighbourhood taken over 12 records rather than the 15 used by the probe."
* Reason: Audit 3; the two implementations differ and score 0.6925 and 0.6818 at window level.

### A17. Section 3.11 and 4.9 (pointers; sample)
* Section 3.11, fourth paragraph. Current: "each applied to every snapshot of every window at
  evaluation time only." Proposed: "each applied to every snapshot of each evaluated window at
  evaluation time only, over the sample of test windows described in Section 4.9."
* Section 4.9, Table 4.8 paragraph. Current: "stood at 0.8269 at the trained value of 0.6".
  Proposed: "stood at 0.8269 at the trained value of 0.6, matching the unperturbed baseline to
  within 0.001 although the edges are rebuilt rather than read from disk".
* The second pointer ("their interpretation in Section 4.9 reflects that") is already
  satisfied by "both of which simulate incomplete operational logging rather than adversarial
  interference"; no change.
* Reason: brief items 4 and E8. (If the sweeps are re-run under E8, Section 4.9's sample
  statement changes with them.)

### A17b. Sections 3.11 and 4.9 (what the embedding-noise and edge-loss perturbations do)
* Section 4.9, paragraph after Table 4.7. Current: "which is what a detector reading
  structure rather than content would be expected to show, since noise added to the embedding
  dimensions perturbs the similarity matrix from which edges are constructed and therefore
  degrades the graph progressively rather than destroying a content signal at a point."
  Proposed: "although the perturbation leaves the graph itself untouched: edges, their weights
  and the structural node features are carried through unchanged, so the decline measures the
  detector's sensitivity to noise in the embedding columns it reads directly, through the
  memory module and the attention layers, and not a degradation of the similarity graph."
* Section 3.11, fifth paragraph, after "leaving the three behavioural dimensions untouched":
  add "; the noise is drawn independently at every snapshot, and the edges are not rebuilt
  from the noisy embeddings".
* Section 3.11, sixth paragraph. Current: "The fourth simulates incomplete indexing by
  dropping each edge independently with probability r". Proposed: "The fourth simulates
  incomplete indexing by dropping each directed edge entry independently with probability r,
  at every snapshot, and recomputing the structural node features from the edges that remain".
* Reason: `noise_transforms.py`. `add_embedding_noise` changes `x[:, :384]` only; the
  threshold and edge-loss perturbations are the ones that rebuild edges and refresh the
  structural columns.

### A18. Section 3.13 (generated content; three strategies)
* First paragraph. Current: "All interaction content, memory records and attack trajectories
  were synthetically generated, either by parameterised topic clusters during development or
  by a language model prompted to produce fictional conversational content for the
  trajectories reported in Chapter Four. Where an external language model interface was used
  for generation, only the provider's published terms of service applied, and no confidential
  or proprietary material was transmitted to that interface."
  Proposed: "The memory records of the corpus reported in Chapter Four are unmodified entries
  from the StrategyQA knowledge base released for research with AgentPoison (Chen et al.,
  2024), and the trajectories into which they are placed were generated by the simulator.
  Corpora used during development were generated by parameterised topic clusters and by a
  language model prompted to produce fictional content; where an external language model
  interface was used, only the provider's published terms of service applied, and no
  confidential or proprietary material was transmitted to it."
* Second paragraph. Current: "as all three simulated strategies are already documented in the
  published literature mentioned in Section 3.6." Proposed: "as the simulated strategy is a
  simplified form of the multi-record injection documented in the literature reviewed in
  Section 2.3."
* Reason: brief item 3.

### A19. Section 3.12 (level of the acceptance criterion)
* Current: "The first required a graph-level area under the receiver operating
  characteristic curve of at least 0.85."
* Proposed: "The first required a graph-level area under the receiver operating
  characteristic curve of at least 0.85 at window level, the granularity at which validation
  performance was monitored during training."
* Reason: brief item 13 and E7. The configuration file states 0.85 without a level; the only
  validation figure computed during development is the window-level one, so this records what
  was done. Section 5.3 is then rewritten like for like in B4.

### A20. Section 4.3, Section 5.5, Section 5.8, Section 6.4 (node vs sequence ranking)
* Section 4.3, third paragraph. Current: "and it exceeds its own trajectory-level ranking by
  a wide margin." Proposed: "and it exceeds the trajectory-level ranking of the same windows,
  0.8261, by a wide margin."
* Section 5.5, fourth paragraph. Current: "at 0.9087 against 0.8122 on unperturbed data."
  Proposed: "at 0.9087 against 0.8261 for trajectory-level ranking of the same windows."
* Section 5.8, second paragraph from end. Current: "at 0.9087 against 0.8122". Proposed:
  "at 0.9087 against 0.8261 at window level".
* Section 6.4: no number is quoted; no change.
* Reason: brief item 14. (Numbers move with C.)

### A21. Table 4.5 (parameter counts; difference column)
* "Gated readout | ... | Not recorded" to "671,686".
* "Temporal components removed, mean pooling | ... | Not recorded" to "297,922".
* Caption, append: "The difference column is the point difference; the text quotes the
  paired bootstrap mean, which for the temporal ablation is 0.2517 with an interval of
  [0.2049, 0.2995]."
* Reason: brief item 8; Audit 3.

### A21b. Table 4.4 and Section 4.3 (static GAT record-level ranking)
* **Updated 23 Oct:** the numbers below are superseded by C8 (main corpus: 0.7445 for the static
  model, and the matched-cluster comparison).
* Table 4.4, static graph attention network, last column. Current: "Not measured". Proposed:
  "0.8832".
* Section 4.3, third paragraph, after "Node-level ranking is, notably, the strongest single
  figure the detector produces": add "although the static graph model, which sees only the
  final snapshot, ranks records at 0.8832, so most of that ranking does not depend on the
  temporal components".
* Reason: measured from `run_burst2k_static_attn/phase2_best.pt` (`static_attn_test.npz`;
  node AP 0.3310). The static model has no memory module, so batch size does not affect it.

### A22. Section 4.7 (significance claim supported; add the intervals)
* Current: "Replacing attention pooling with mean pooling reduces performance to 0.6207, a
  difference of 0.1915."
* Proposed: "Replacing attention pooling with mean pooling reduces performance to 0.6207, a
  paired difference of 0.1917 with an interval of [0.1456, 0.2377]."
* Current: "Removing the four structural node features leaves the detector at chance, at
  0.5078 with a 95% interval of [0.4550, 0.5623]." Add: "a paired difference of 0.3049 with
  an interval of [0.2424, 0.3645]".
* Reason: "All three effects are significant" was asserted without intervals for two of them;
  recomputed with the thesis's own paired bootstrap.

### A23. Section 4.10 and Table 4.11 (profile of the wrong model)
* Replace every figure in Section 4.10 and Table 4.11 with `compute_profile.json` in this
  folder (re-measured on the reported checkpoint and on ap_burst2k test windows), and the
  checkpoint size with 2.57 MB in Sections 3.8, Table 3.5 and 4.10.
* Section 4.10, last paragraph: delete "Processing several windows together corrupts the
  persistent memory state maintained by the detector ... A deployment using the record-level
  output, or acting on a threshold instead of on a ranking, must therefore operate at a batch
  size of 1 and take the corresponding throughput of 19.32 windows per second as the figure
  of record. The batched figure is reported here because it is the measured cost of the
  configuration, not because that configuration is recommended for the full detector as
  specified." The code now gives each window its own memory slots, so batched inference is
  exact.
* Re-measured values (`compute_profile.json`; 200 seeded test windows, forward pass only,
  per-window memory, Quadro P2000):

  | Measurement | Batch size 1 | Batch size 4 |
  |---|---|---|
  | GPU mean latency (ms) | 87.35 | 131.54 |
  | GPU 95th percentile latency (ms) | 104.32 | 147.68 |
  | GPU throughput (windows per second) | 11.45 | 30.41 |
  | CPU mean latency (ms) | 57.37 | 105.90 |
  | CPU throughput (windows per second) | 17.43 | 37.77 |
  | Peak GPU memory (MiB) | 22.78 | 31.81 |

  Text that quotes "approximately 52 milliseconds" (Sections 4.10, 5.8, 6.3, 6.4) becomes
  "approximately 90 milliseconds on the graphics processing unit and 60 on the central
  processing unit". The conclusion that cost is not the binding constraint at one snapshot per
  10 interactions still holds. The sentence "Peak memory occupancy was identical at both batch
  sizes because the dominant allocation is the node memory bank" is not supported (22.78 vs
  31.81 MiB) and should go.
* Reason: Audit 3; the old profile is of a 667,331-parameter model on the earlier benchmark,
  and its script no longer exists.

### A24. Section 2.5 reference (Li et al., 2026) and the reference list
* Replace the en dash with a hyphen in the 19 page ranges that use one: Cai et al. (2021),
  Cho et al. (2014), Ding et al. (2019), Greshake et al. (2023), Hevner et al. (2004), Ilse et
  al. (2018), Jin et al. (2024), Khosla et al. (2020), Kumar et al. (2019), Lin et al. (2017),
  Liu et al. (2008), Lo et al. (2022), Pareja et al. (2020), Park et al. (2023), Peffers et al.
  (2007), Reimers and Gurevych (2019), Sankar et al. (2020), Schölkopf et al. (2001), Zhong et
  al. (2024). Mechanical; can be done directly.
* Dong et al. (2025): add the proceedings details. "In Advances in Neural Information
  Processing Systems (Vol. 38, pp. 52119-52153). Curran Associates.
  https://doi.org/10.52202/085713-1554" (from the NeurIPS 2025 proceedings listing at
  proceedings.com; confirm the DOI resolves before inserting).
* Du (2026): add "https://arxiv.org/abs/2603.07670" in place of the bare identifier, to match
  the other arXiv entries.
* Li et al. (2026): authors and article number confirmed (Yuanlin Li, Li Feng, Zonghai Zhu,
  Huanlai Xing; Ad Hoc Networks, Article 104292). Volume and DOI not found in public listings;
  take them from the ScienceDirect article page.
* Reason: brief item 18.

### A25. Section 6.3 and Abstract (three vs four contributions)
* Section 6.3. Current: "The study makes four contributions, corresponding to the four
  objectives stated in Section 1.4, and they are stated here in order of how far each
  transfers beyond the present study." Proposed: "The study makes four contributions, which
  answer the four objectives stated in Section 1.4 and are stated here in order of how far
  each transfers beyond the present study: the first answers Objective IV, the second
  Objective I, the third Objective II and the fourth Objective III."
* Abstract. Current: "The contribution is threefold, comprising a formal representation of
  agent memory, a validated benchmark together with the protocol that validates it, and an
  attribution methodology." Proposed: "The contribution is fourfold, comprising an attribution
  and benchmark validation methodology, a formal representation of agent memory with a
  benchmark that satisfies that methodology, the detector, and a comparison against
  implemented alternatives."
* Reason: brief item 15. (Revise the Abstract in your own voice afterwards, brief item 21.)

### A26. Section 4.3 (five evaluations: no artefact)
* Current: "Across five independent evaluations of the retained checkpoint the graph-level
  area under the curve ranged from 0.8259 to 0.8264, a spread of 0.0006 with a standard
  deviation of 0.0002, and the reported figure lies within that range."
* Proposed, if Chapter 4 adopts the as-specified configuration: "Five independent
  evaluations of the retained checkpoint returned the same graph-level area under the curve
  to every digit reported, so evaluation contributes no run-to-run variation."
  Also in the same paragraph, delete "since inference on the graphics processing unit is not
  bitwise reproducible. The scatter operations underlying attention pooling accumulate in
  nondeterministic order." The earlier fourth-decimal variation came from windows sharing
  memory slots (Audit 1.2), not from attention pooling.
* If Chapter 4 keeps the batched figures instead, delete the sentence: it has no artefact.
* Reason: brief E10. Nothing in the repository or its history records the five values. The
  re-measurement (`base_test_repeat1..5.npz`, per-window memory) gave 0.8248917939455518 in
  every run, as did batch-1 scoring with and without the cache.

### A27. Section 6.5, first paragraph (target query)
* Current: "The fourth is retention of the target query the simulator constructs for each
  injection, which would permit retrieval success to be measured"
* Proposed: "The fourth is the construction of a target query for each injected cluster,
  which would permit retrieval success to be measured"
* Reason: the ap_burst2k generator (`agentpoison_sequences.py`) constructs no target query;
  that logic belonged to the superseded LLM generator. E6 therefore needs a new design, not a
  retention fix.

### A28. Front matter (brief items 19, 20)
* Your decision: confirm whether the Declaration, Certification, Acknowledgements,
  Abbreviations and the Lists of Tables and Figures were dropped on purpose; change the title
  page date if the submission month is not September; update all fields last (select all, F9).

---

## B. Apply now with interim numbers (replace with E2 means when available)

### B1. Chapter 4, new comparison against the oracles (E1)
Add to Table 4.4, or as a Table 4.4 continuation headed "Engineered-feature oracles", the
following sequence-level rows. These are the detector as specified (batch 1;
`oracle_comparison_b1.json`), single seed:

| Method | Sequence AUROC [95% interval] | AP | Detector minus method, AUROC | Detector minus method, AP |
|---|---|---|---|---|
| Logistic regression, final-state (6) | 0.4650 [0.4146, 0.5190] | 0.3872 | +0.354 [+0.298, +0.409] | +0.422 [+0.365, +0.480] |
| Logistic regression, temporal (7) | 0.7934 [0.7526, 0.8311] | 0.7108 | +0.027 [-0.021, +0.078] | +0.102 [+0.039, +0.169] |
| Logistic regression, all (13) | 0.8030 [0.7630, 0.8397] | 0.7203 | +0.017 [-0.034, +0.070] | +0.093 [+0.030, +0.162] |
| Gradient boosting, all (13) | 0.8204 [0.7832, 0.8542] | 0.7631 | -0.000 [-0.051, +0.048] | +0.052 [-0.006, +0.109] |
| Logistic regression, nine features, final snapshot | 0.4951 [0.4449, 0.5439] | 0.4005 | +0.324 [+0.266, +0.382] | +0.409 [+0.349, +0.467] |
| Proposed detector | 0.8194 [0.7788, 0.8579] | 0.8142 | | |

Proposed paragraph for Section 4.6, after the static-model paragraph:

"Table 4.4 also reports the engineered-feature oracles of Section 3.10 at sequence level,
together with gradient boosting fitted to the same thirteen features with its settings chosen
on the validation partition. The detector separates clearly from every model restricted to the
final state, and from the nine-feature regression applied to each sequence's last snapshot. It
does not separate from any model that reads the hand-built temporal summaries. On area under
the curve it ties the thirteen-feature regression and gradient boosting alike, the paired
intervals spanning zero. On average precision it leads the regression, by 0.093 with an
interval of [0.030, 0.162], and leads gradient boosting by 0.052, with an interval that just
includes zero. The detector therefore recovers, from the graph and without being told what to
look for, a signal that an oracle hand-built from knowledge of the generator recovers equally
well, and it does not exceed that oracle."

### B2. Sections 5.4, 5.5, 5.6, 6.3, 6.6 (overreach)
* Section 5.6. Current: "a detector built on the closed gap outperforms every implemented
  alternative on a corpus where the obvious shortcuts have been measured and excluded."
  Proposed: "a detector built on the closed gap outperforms every implemented alternative that
  is not given hand-built temporal summaries, and matches those that are, on a corpus where
  the obvious shortcuts have been measured and excluded."
* Section 5.5, last paragraph. Current: "The honest statement is that the detector ties the
  oracle on whole-ranking quality and beats it where an analyst would look". Proposed: "The
  statement the evidence supports is that the detector ties the linear oracle on
  whole-ranking quality and leads it where an analyst would look, and that against gradient
  boosting on the same features even that lead is not established".
* Section 6.3, fourth contribution. Add at the end: "Against oracles built from knowledge of
  the generator the detector ties on area under the curve, so the comparison establishes that
  the trajectory signal can be learned from the graph, not that the detector exceeds what an
  informed analyst could engineer."
* Section 5.4, first paragraph, after "The benchmarks reported in Table 4.4 fall into an order
  that locates where the detection signal lies.": add "The oracles of Section 3.10 sit at the
  top of that order with the detector, which ties them on area under the curve."
* Section 6.6: no sentence claims superiority over the oracle; no change beyond C.
* Reason: brief items 12 and E1.

### B3. Abstract (oracle tie, brief items 12, 21)
* Add after the static-model sentence: "Against oracles built from hand-crafted temporal
  features with knowledge of the attack, the detector tied on area under the curve."

### B4. Section 5.3 (acceptance, like for like)
* Current: "The acceptance criterion of 0.85 stated in Section 3.12 was assessed on the
  validation partition, where the retained checkpoint attained 0.8979 and the criterion was
  met. The sequence-level test figure of 0.8122 falls below that threshold, and the two are
  not directly comparable, since the criterion was specified at window level and on a
  different partition. Both are reported here rather than the more favourable one alone."
* Proposed: "The acceptance criterion of 0.85 stated in Section 3.12 was assessed at window
  level on the validation partition, where the retained checkpoint attained 0.8979 and the
  criterion was met. Measured the same way on the test partition the detector attains 0.8261,
  so the criterion is not met on held-out data; at sequence level the corresponding figures
  are 0.8556 and 0.8122. The fall from validation to test is a matter of composition rather
  than of generalisation. Restricted to attacks of nine or more records, sequence-level
  ranking is 0.9028 on validation and 0.9131 on test, and attacks of eight or fewer are close
  to chance on both partitions; the test partition simply holds a larger share of them,
  22.5 per cent against 15.8 per cent."
  (Figures as reported; the as-specified equivalents are 0.8845 and 0.8249 at window level,
  0.8525 and 0.8194 at sequence level. Use whichever configuration Chapter 4 adopts.)
* Reason: brief item 13 and E7.

### B5. Section 4.5 (size, duplication and burst length)
* Reinstate a caution after "Performance depends strongly on size.": "Size is not varied in
  isolation in this corpus. A cluster of more than eight records necessarily repeats entries,
  because both the attack and its control draw from the eight nearest neighbours of one entry,
  and an attack occupies one interaction per record, so size, duplication and burst length
  move together."
* At sequence level the detector is at chance for every size from 5 to 8 (0.45 to 0.56) and
  steps to 0.78 at 9, the first size with repeated entries (`attack_size_by_k_b1.json`).
* E4 grid result (Audit 1.4), for a new paragraph in Section 4.5: "Varied one at a time on
  freshly generated sequences, burst length and duplication both matter. The same records
  spread over 80 interactions are detected at chance whatever their number, so the detector
  does respond to the rate of arrival. Clusters of 16 to 20 records arriving in a burst are
  detected at 0.96 when they repeat entries and at 0.63 to 0.70 when they do not, so the
  strongest results in Table 4.3 depend on repeated entries rather than on size alone."
  Add the caveat that the detector was trained on clusters containing repeats.

---

## C. Results from the freeze (`experiments/audit/RESULTS_FREEZE.md`, 23 October 2026)

Every number in this group is copied from RESULTS_FREEZE.md, which names the file each comes
from. The main benchmark is now **ap_burst2k_nodup** (no repeated entries in any cluster);
five training seeds (1 to 5) per configuration. Where groups A or B quote an older number,
these replace it.

### C0. The benchmark change (Sections 3.6, 3.7, 3.15, 4.5, 5.2, 5.7, 6.2, Abstract)
* Section 3.6, add after the sentence on the matched control (replaces the sentence proposed
  in A13): "Each cluster, in the attack and in its control alike, is drawn without replacement
  from the nearest neighbours of one randomly chosen entry, at least eight and at least as many
  as the cluster requires, so no cluster repeats an entry. An earlier version of the corpus drew
  from eight neighbours with replacement, which made every cluster of more than eight records
  repeat entries; Section 4.5 reports why that version was replaced."
* Section 3.7: name the corpus ap_burst2k_nodup wherever the corpus is named, and state the
  command in RESULTS_FREEZE section 1 in the reproducibility material (appendix or repository
  note), not in the prose.
* Section 4.5, a new paragraph reporting R1 (text below in C5).
* Every chapter that quotes 0.8122 (now 0.6614), 0.2517 (now 0.1206), 0.9087 (now 0.7572) or
  the 0.5735 and 0.9554 size bands follows the items below.

### C1. Headline (Abstract, 4.3, Table 4.1, 4.11, 5.3, 6.2, 6.3, 6.6)
* "0.8122 ... [0.7707, 0.8532] ... average precision of 0.8185" becomes "0.6614 (standard
  deviation 0.0144 across five training seeds; 95% interval [0.6166, 0.7054]) ... average
  precision of 0.6115".
* Table 4.1, window level (sequence level in brackets): graph AUROC 0.7044 (0.6614); graph AP
  0.5649 (0.6115); F1 at 0.5 0.4833; FPR at 90% TPR 0.7299; node AUROC 0.7572; node AP 0.1819.
  Delete the "Best graph F1-score (threshold 0.2129)" row.
* Section 4.3, the paragraph on F1 ("the default threshold costs very little here"): the
  best-F1 comparison is no longer reported; keep the sentence on the cost of high recall with
  0.7299 in place of 0.5757.
* Text quoting 0.9087 for node ranking becomes 0.7572; 0.5662 becomes 0.1819; 0.8261 becomes
  0.7044; 0.5757 becomes 0.7299.

### C2. Seeds and noise floor (Abstract, 3.3, 3.14, 4.1, 4.3, 4.7, 5.7, 6.5)
* Abstract: "two training seeds" becomes "five training seeds".
* Section 4.3, stability paragraph, proposed: "Five training runs under seeds 1 to 5 attained a
  sequence-level area under the curve of 0.6614 with a standard deviation of 0.0144, and
  repeated evaluation of one checkpoint returned the same figure to every digit reported, so
  run-to-run variation comes from training alone. Differences smaller than about twice that
  standard deviation, 0.03, are treated throughout this chapter as noise."
* Sections 3.14, 4.7, 5.7: "0.02" becomes "0.03" (twice 0.0144), described as twice a measured
  standard deviation. Section 3.14: "Two runs differing only in random seed produce
  sequence-level figures 0.0018 apart" becomes "Five runs differing only in random seed have a
  standard deviation of 0.0144".
* Section 5.7: replace the sentences from "The limitation that remains is that two seeds are
  not a variance." to "the cheapest remaining improvement." with "Each configuration was
  trained under five seeds, so the noise floor of approximately 0.03 used throughout Chapter
  Four is twice a measured standard deviation."
* Section 6.5: delete the direction on a third training seed.
* Section 4.1: replace "a second training run under seed 1 is reported in Section 4.3" with
  "results are means over five training seeds, 1 to 5".

### C3. Table 4.5 and Section 4.7 (components, capacity, readout)
Replace Table 4.5 with RESULTS_FREEZE Table 4.5. In short (difference = detector minus):
memory kept, aggregator removed 0.6515 (+0.0098, not significant); aggregator kept, memory
removed 0.5841 (+0.0773, significant); both removed 0.5408 (+0.1206); both removed and widened
to 670,233 parameters 0.5410 (+0.1203); mean pooling 0.6187 (+0.0427, interval
[-0.0043, 0.0897], not significant); structural features removed 0.5396 (+0.1218). The gated
readout row is deleted (single seed, earlier corpus).

* Section 4.7, first paragraph, proposed: "Removing the four structural node features leaves
  the detector at chance, at 0.5396. Removing the node memory module and the temporal
  aggregator together reduces performance to 0.5408, a paired difference of 0.1206 with a 95%
  interval of [0.0762, 0.1626]. The two components do not contribute equally: without the
  node memory the detector attains 0.5841, a loss of 0.0773, while without the temporal
  aggregator it attains 0.6515, a loss of 0.0098 that lies within the noise floor. The
  temporal capacity therefore sits almost entirely in the per-node memory. Replacing attention
  pooling with mean pooling reduces performance to 0.6187, a difference of 0.0427 whose
  interval includes zero."
* Section 4.7, the paragraph beginning "The mean-pooling variant isolates the readout exactly":
  keep the parameter argument (8,321) and replace "An 8,321-parameter difference produces a gap
  of 0.1915 in sequence-level ranking. This is the direct confirmation of the argument ..."
  with "On this corpus the 8,321 parameters of attention pooling buy a gain of 0.0427 that the
  interval does not distinguish from zero, so the argument of Sections 2.6 and 3.8 that mean
  pooling attenuates a small anomalous subgraph is not confirmed here."
* Section 4.7, the concession paragraph: replace "and the counter-explanation that the variant
  is simply smaller remains available" with "and a static variant widened to 670,233
  parameters attains 0.5410, no better than the smaller one, so the counter-explanation that
  the static model is simply smaller is not supported". Delete "That counter-explanation does
  not apply to the mean-pooling result, where the difference is 8,321 parameters and the gap is
  0.1915, which is why that comparison carries the weight of the component argument."
* Section 4.7, the paragraph on structural features: "costs 0.2517" becomes "costs 0.1206".
* Sections 5.5 and 6.3: statements that attention pooling is load-bearing or confirmed by
  direct measurement must be withdrawn or qualified to "not confirmed on the final corpus".
* Section 6.5: delete the direction asking for a node-memory-only ablation; Section 3.10 gains
  one sentence describing the three new variants.

### C4. Table 4.2 and Section 4.4 (latency)
Replace Table 4.2 with RESULTS_FREEZE Table 4.2 (5% target): detector realises 4.79% and
detects 81.0 of 200 (40.5%), median latency 4, mean 4.75, pre-onset alarms 0.064; regression
realises 4.48% and detects 64 (32.0%), median 7, mean 6.73. Caught within 0, 2, 5 and 10
snapshots: detector 2.3%, 6.7%, 28.4%, 39.1%; regression 1.5%, 6.0%, 14.0%, 25.5%; differences
+0.008 [-0.014, 0.029], +0.007 [-0.031, 0.046], +0.144 [0.079, 0.211], +0.136 [0.059, 0.217].

* Section 4.4, second paragraph: the argument about the regression's shorter median no longer
  applies. Proposed replacement for its first three sentences: "The detector detects 81 of the
  200 poisoned trajectories against the regression's 64, and among those it detects, it does
  so sooner, with a median latency of four snapshot intervals against seven." Replace "on that
  measure the detector leads at every horizon by a margin whose interval excludes zero" with
  "on that measure the detector leads from five snapshots onward, while within the first two
  snapshots neither catches more than seven per cent and the two do not differ". Replace the
  ten per cent sentence with "At a ten per cent target the regression detects more trajectories
  than the detector, 119 against 105, though more slowly, with a median of seven intervals
  against three."
* Abstract, 4.11, 5.3, 6.3, 6.6: "67.5 per cent ... against 30.0 per cent" becomes "40.5 per
  cent ... against 32.0 per cent"; "median detection latency of two snapshot intervals" becomes
  "four snapshot intervals".

### C5. Table 4.3 and Section 4.5 (attack size; the duplicates finding)
Replace Table 4.3 with RESULTS_FREEZE Table 4.3 (size-matched, sequence level): 5 to 8 records
0.618; 9 to 12, 0.608; 13 to 16, 0.649; 17 to 20, 0.761.

* Section 4.5, first paragraph, proposed figures: "Attacks of sixteen records or fewer are
  detected at 0.61 to 0.65, and attacks of seventeen or more at 0.761." Replace "each row
  evaluates the poisoned sequences falling within one size band together with all 3,300
  windows drawn from clean sequences" with "each row evaluates the poisoned sequences in one
  size band against the clean sequences whose matched benign cluster falls in the same band,
  the comparison the matched-control construction is designed for".
* Section 4.5, new paragraph (R1): "An earlier version of the corpus drew each cluster from
  eight neighbours with replacement, so every cluster of more than eight records repeated
  entries, in the attack and its control alike. On that version the detector attained 0.8026,
  but three counts of repeated records, fitted by logistic regression, attained 0.7635 on their
  own and 0.8495 together with the thirteen engineered features, above the detector. The
  detector was therefore reading bursts of repeated entries at least as much as the rate of
  arrival. Without repeated entries those counts are at chance, at 0.5049, and the detector
  attains 0.6614; its advantage over a static model falls from 0.2493 to 0.1206 and remains
  significant. The corpus without repeated entries is the one reported throughout this chapter."
* Section 4.5, the caution sentence proposed in B5 still applies, without the clause about
  repeated entries.
* Every later mention of 0.5735 and 0.9554 (4.11, 5.3, 5.7, 6.2, 6.6) becomes 0.618 and 0.761.
  Sentences saying detection is "close to chance below nine records" become "weak at every
  size up to sixteen records (0.61 to 0.65)".

### C6. Oracle comparison (B1 replaced)
Use RESULTS_FREEZE section 4 for Table 4.4's oracle rows. The B1 paragraph changes, proposed:
"The detector separates clearly from every model restricted to the final state, and from the
nine-feature regression applied to each sequence's last snapshot. It does not separate from any
model that reads the hand-built temporal summaries, on area under the curve or on average
precision: against the thirteen-feature regression the differences are -0.025 and +0.008, and
against gradient boosting -0.003 and +0.050, with every interval spanning zero. The detector
therefore recovers from the graph, without being told what to look for, the signal that an
oracle hand-built from knowledge of the generator recovers, and it does not exceed it."
* Section 5.5, last paragraph: the AP-lead argument no longer holds. Replace from "The detector
  leads it on average precision" to the end of the paragraph with "The detector ties it on area
  under the curve and on average precision alike, so the honest statement is that the detector
  matches the oracle without being given its features, and does not beat it."
* B2's Section 5.5 wording ("leads it where an analyst would look") is withdrawn for the same
  reason.

### C7. Acceptance (B4 replaced)
Proposed for Section 5.3: "The acceptance criterion of 0.85 stated in Section 3.12 was not met.
At window level the detector attains 0.7001 on the validation partition and 0.7044 on the test
partition; at sequence level 0.6538 and 0.6614. Validation and test agree, so the shortfall is
a property of the task as constructed, not of overfitting to the validation partition."

### C8. Table 4.4, record level, and node-level claims (A21b replaced)
* Table 4.4: unsupervised rows become Isolation Forest 0.5748 [0.5219, 0.6267], AP 0.4870,
  record level 0.5108 (AP 0.0273); One-Class SVM 0.5210 [0.4674, 0.5715], AP 0.4747, record
  level 0.5000 (AP 0.0257); static GAT 0.5408 [0.4916, 0.5909], AP 0.4759, record level 0.7445;
  detector 0.6614 [0.6166, 0.7054], AP 0.6115, record level 0.7572.
* Section 4.3, third paragraph: "Node-level ranking is, notably, the strongest single figure
  the detector produces" becomes "Node-level ranking, at 0.7572, mostly measures membership of
  a dense cluster, which a single snapshot shows: the static model reaches 0.7445. Measured
  against the records of the matched benign clusters, which are as dense as an attack, the
  detector ranks injected records at 0.6611 and the static model at 0.5461, the same gap as at
  sequence level."
* Sections 5.5, 5.8 and 6.4: delete the claims that node-level ranking is "the stronger figure"
  and "the more robust"; under the sweeps node ranking falls from 0.7572 to 0.6886 at the
  heaviest noise, against 0.7044 to 0.5989 for window-level ranking, so it is the more robust
  of the two but not the stronger.

### C9. Tables 4.6 to 4.11 and Table 3.7
* Table 4.6: structural 0.6341 (AP 0.4218); pooled content 0.4967 (0.3058); combined 0.6309
  (0.4205); per record 0.5143 (0.0267, base rate 0.0256); detector window level 0.7044. Section
  4.8: "leaves a distance of 0.1443 to the detector's window-level figure of 0.8261" becomes
  "leaves a distance of 0.0703 to the detector's window-level figure of 0.7044"; "the ablation
  reported in Section 4.7 measures that capacity independently at 0.2517" becomes "at 0.1206".
* Tables 4.7 to 4.10: replace with RESULTS_FREEZE Tables 4.7 to 4.10 (full test partition,
  five seeds, three draws). Section 4.9's sample paragraph becomes: "Each sweep was evaluated on
  the full test partition of 5,500 windows, for each of the five trained detectors, with three
  random draws of every stochastic condition." Section 4.9 conclusions: embedding noise remains
  the largest sensitivity (0.7044 to 0.5989 at 0.30); the threshold sweep peaks at 0.5 to 0.6
  and falls to 0.6222 at 0.8, so the choice of 0.6 matters more than the text claims ("every
  value tested lies within 0.043 of it" becomes "values from 0.5 to 0.6 lie within 0.003 of
  each other, and 0.8 costs 0.082"); metadata corruption and edge loss each cost 0.0255 at 0.30.
  Item A17b still applies to the explanation of embedding noise.
* Table 4.11: replace with RESULTS_FREEZE Table 4.11 (89.11 ms and 11.22 windows per second on
  the GPU at batch 1, 30.39 per second at batch 4; 56.15 ms on the CPU; 22.75 to 32.29 MiB).
  "approximately 52 milliseconds" becomes "approximately 90 milliseconds on the graphics
  processing unit and 60 on the central processing unit"; this replaces the values in A23.
* Table 3.7 and Section 3.15: per-record probe 0.5143 (AP 0.0267, base rate 0.0256); pooled
  0.4967; position alone 0.5950 (test split); final-state 0.5514 [0.5043, 0.6006]; node count
  0.5000; temporal 0.6594. The final-state verdict becomes "near chance (the interval just
  excludes 0.5)" and the text "Every leakage control sits at chance" becomes "Every leakage
  control sits at or near chance".

### C10. Figures 4.1 to 4.3
Replace with `experiments/rebuild/figures/` (seed 1, the median seed; sequence
poisoned_placement_0756, chosen by the rule in RESULTS_FREEZE section 7). Section 4.3 figure
text, proposed: "The score distribution shows the 188 benign records of the final window and
the 12 injected records, with an operating threshold of 0.440 selected at a five per cent
false-alarm rate on the validation partition. The detector flags 37 records, recovering 11 of
the 12 injected ones and attaching 26 benign records." Figure 4.3 text: onset at snapshot 12,
first alert at snapshot 17 (operating point 0.944). The earlier example, poisoned_placement_0463,
is missed by this model and can illustrate the limitation in Section 5.7
(`experiments/rebuild/figures_missed_example/`).

### C11. Figures 4.4 to 4.7
Replace with `experiments/rebuild/figures/fig_4_4` to `fig_4_7` (main corpus, five seeds, three
draws, full test partition; mean +/- SD error bars). Each caption should add "mean and standard
deviation over five training seeds and three random draws". The figures add a dashed
sequence-level line to the three window and record-level lines of the earlier versions.


---

## D. Results from freeze 2 (`experiments/audit/RESULTS_FREEZE_2.md`, 3 November 2026)

Every number in this group is copied from RESULTS_FREEZE_2.md ("FZ2", with its section), which
names the file each comes from, or from `experiments/IMPROVEMENT.md` ("IMP"). The corpus is
unchanged (ap_burst2k_nodup). "The detector" is the temporal graph neural network; "the
combined detector" is the detector and the static graph model with their ranks averaged, the
configuration chosen in advance on the validation partition (FZ2 section 3). Freeze 1 figures
that still stand (Table 3.7, the unsupervised rows of Table 4.4, the probes, the R1 paragraph of
C5) keep C's wording. Current text is quoted from `MosesWinboodAyandauTGNN-3.docx` as it stands;
where an A, B or C item has already changed a sentence, apply D to the changed sentence.

### D0. Decisions for Moses and the supervisor, before applying D
* **Title.** "DETECTION OF PERSISTENT MEMORY POISONING IN LARGE LANGUAGE MODEL AGENTS USING
  TEMPORAL GRAPH NEURAL NETWORKS" stays accurate as a description of what was built and tested,
  but the results show the temporal components add nothing measurable. Options: keep it, and let
  Chapters 4 to 6 state the finding (no change proposed here); or "... USING GRAPH NEURAL
  NETWORKS AND ARRIVAL-ORDER FEATURES". A registered title may need the department's approval.
* **Aim (1.3), objectives (1.4), research questions (1.5): no change proposed.** They describe
  what was built and asked; D changes the answers. Objective IV and research question IV already
  ask how far performance is attributable to temporal modelling "as opposed to simpler signals",
  which is the question D answers.

### D1. Abstract
* Second paragraph. Current: "in which nodes are stored records carrying semantic and
  behavioural attributes and edges encode semantic similarity, so that poisoning becomes a
  detectable change in the trajectory of that graph." Proposed: "in which nodes are stored
  records carrying semantic and behavioural attributes, together with measures of how closely
  each record agrees with the records that arrived beside it, and edges encode semantic
  similarity and order of arrival, so that poisoning becomes a detectable change in the
  trajectory of that graph."
* Second paragraph. Current: "was trained to classify trajectories and to rank individual
  records." Proposed: "was trained to classify whole trajectories and to rank individual
  records, beside a static graph model reading the final snapshot alone."
* Third paragraph (replaces C1's Abstract item and the Abstract parts of C2, C4, C5).
  Replace from "On a held-out partition of 500 sequences" to "two training seeds." with:
  "On a held-out partition of 500 sequences the detector and the static graph model, combined
  by averaging their ranks as fixed in advance on the validation partition, attained a
  sequence-level area under the curve of 0.9221 (standard deviation 0.0089 across five training
  seeds; 95 per cent interval [0.8975, 0.9424]) and an average precision of 0.9080. The temporal
  detector alone attained 0.9088 and the static model alone 0.9156, a difference the interval
  does not distinguish from zero, so the temporal components add nothing measurable once the
  representation records the order of arrival; without the arrival measures the detector
  attained 0.6680. At a matched five per cent false-alarm target the detector caught 64.9 per
  cent of poisoned trajectories against 32.0 per cent for the strongest feature-based baseline,
  with a median detection latency of two snapshot intervals against seven. Both unsupervised
  benchmarks performed at or near chance at record level, establishing that injected records
  are not distributional outliers. The contribution is threefold, comprising a formal
  representation of agent memory that records the order in which records arrive, a validated
  benchmark together with the protocol that validates it, and an attribution methodology. The
  findings rest on simulated data, a single threat model and five training seeds, and the
  arrival measures are specific to an attack that writes its records in a consecutive burst."
  (If A25 has changed the count of contributions, keep its count and amend only the first item.)
  Sources: FZ2 sections 3, 4 (Tables 4.2, 4.5).

### D2. Section 1.8, third paragraph (unit of detection)
Current: "The detector classifies a sequence of 10 memory graph snapshots as arising from
either a healthy or a compromised trajectory." Proposed: "The detector classifies the 20
memory graph snapshots of a trajectory as arising from either a healthy or a compromised
trajectory, and is also read on the snapshots received so far, at eleven points through the
trajectory, to measure how soon it responds." Reason: one window per sequence (FZ2 section 0).

### D3. Sections 3.4 and 3.7 (one window per sequence; readings over time)
* Section 3.4, first paragraph. Current: "Consecutive snapshots were then grouped into
  overlapping sliding windows of 10 snapshots each, so that each sequence yielded 11 windows and
  each window constituted one training or evaluation example. The overlap is deliberate, since
  it allows the detector to be evaluated at every point in a trajectory instead of only at fixed
  boundaries, but it also imposes the partitioning constraint described in Section 3.7."
  Proposed: "The 20 snapshots of a sequence formed a single window, so that each sequence
  constituted one training or evaluation example. So that the detector could be evaluated at
  every point in a trajectory and not only at its end, it was also read on the snapshots
  received so far after each snapshot from snapshot 9 to snapshot 19, which are the end points
  of eleven overlapping windows of 10 snapshots, and these readings impose the partitioning
  constraint described in Section 3.7."
* Section 3.4, second paragraph. Current: "The resulting sequence of 10 snapshot
  representations is then consumed". Proposed: "The resulting sequence of snapshot
  representations, 20 for a whole trajectory, is then consumed".
* Section 3.7, first paragraph. Current: "and each sequence yielded 11 overlapping windows of 10
  snapshots, giving 22,000 window-level examples across the dataset." Proposed: "and each
  sequence constituted one example of 20 snapshots; the eleven readings described in Section 3.4
  give 22,000 reading points across the dataset, at which the detector's response over time is
  measured."
* Section 3.7, third paragraph. Current: "This was necessary because adjacent windows overlap by
  construction, sharing up to 9 of their 10 constituent snapshots, so that partitioning at window
  level would allow information from a test trajectory to reach the training partition through
  neighbouring windows." Proposed: "This was necessary because the readings of one sequence
  overlap by construction, each containing every snapshot of the reading before it, so that
  partitioning at reading level would allow information from a test trajectory to reach the
  training partition."
* Section 3.7, last paragraph. Current: "At window level the proportion of positive examples is
  moderate, standing at 1,769 of 5,500 in the test partition". Proposed: "Across the eleven
  readings the proportion of positive examples is moderate, standing at 1,769 of 5,500 in the
  test partition, and at sequence level it is 200 of 500". In the same paragraph, "although the
  dataset contains 22,000 windows, these are not 22,000 independent observations" becomes
  "although the dataset contains 22,000 reading points, these are not 22,000 independent
  observations".

### D4. Section 3.5, Tables 3.1 and 3.2 (arrival edges and arrival-coherence measures)
* Second paragraph. Current: "Each memory record was encoded as a node carrying a feature vector
  of 387 dimensions." Proposed: "... of 401 dimensions." (Replaces A9's 391.)
* New paragraph after the third paragraph (the one ending "Table 3.1 states the complete node
  feature vector."), proposed: "Ten further attributes describe how closely each record agrees
  with the records that arrived beside it, and were added because the attack writes its records
  in a consecutive burst while the matched clean cluster arrives gradually. For runs of 5, 10, 15
  and 20 consecutive records in order of arrival, each record received the highest mean pairwise
  cosine similarity among the runs of that length containing it, together with that value less
  its median over the records of the snapshot. The remaining two are the mean cosine similarity
  between the record and the records that arrived within 5, and within 10, interactions of it.
  All ten are computed within one snapshot from the embeddings and the order of arrival, and never
  from a label. They are built by hand for the attack studied, and Sections 4.7 and 5.7 report
  what they contribute and what they cost."
* Table 3.1: add a row after the structural rows of A9: "391 to 400 | 10 | Arrival coherence |
  Highest mean cosine similarity over runs of 5, 10, 15 and 20 consecutive arrivals containing
  the record, each with its deviation from the snapshot median, and the mean similarity to the
  records arriving within 5 and within 10 interactions".
* New paragraph after the paragraph ending "in the sensitivity analysis reported in Section 4.9."
  (edges), proposed: "A second relation records order of arrival. Every two records that arrived
  at most 10 interactions apart, one snapshot interval, were joined by an arrival edge whether or
  not their similarity met the threshold, because only a quarter of the pairs within an attack
  burst exceed 0.6 and a similarity graph therefore shows most of a burst as unconnected. Every
  edge carries two attributes, the cosine similarity of the pair and a flag that marks arrival
  edges." Source: IMP step 1 and step 3; `dataset.with_arrival_edges`.
* Next paragraph. Current: "The similarity value is continuous and was retained as an edge
  attribute supplied to the graph attention layers". Proposed: "The similarity value is
  continuous and was retained, with the arrival flag, as an edge attribute supplied to the graph
  attention layers".
* Table 3.2: add a row "Arrival edges | Between every two records that arrived at most 10
  interactions apart, carrying their cosine similarity and an arrival flag", and change the
  "Edge weight" row to "Cosine similarity value and arrival flag, retained as edge attributes".
* Labelling paragraph. Current: "The graph-level label of a window was taken from the final
  snapshot in that window". Proposed: "The graph-level label of a window, and of each reading,
  was taken from its final snapshot".

### D5. Section 3.8 and Table 3.5 (detector)
* First paragraph. Current: "the resulting sequence of 10 representations is consumed by the
  aggregator". Proposed: "the resulting sequence of representations, 20 for a whole trajectory,
  is consumed by the aggregator". Current: "which raises the per-node metadata width from three
  to seven." Proposed: "which raises the per-node metadata width from three to seven, and the ten
  arrival-coherence attributes raise it to seventeen." Current: "The complete network contained
  670,915 trainable parameters, distributed across components as stated in Table 3.5, and the
  trained checkpoint occupied 2.56 megabytes. Disabling the four structural features reduces the
  count to 667,331". Proposed: "The complete network contained 680,515 trainable parameters,
  distributed across components as stated in Table 3.5, and the trained checkpoint occupied 2.61
  megabytes. Disabling the four structural features reduces the count to 676,931". Source: FZ2
  Tables 4.5 and 4.11 (2.607 MiB).
* Second paragraph. "391-dimensional" becomes "401-dimensional"; "519-dimensional input" becomes
  "529-dimensional input".
* Third paragraph. Current: "Clearing the bank at window boundaries constrains the history
  available to 10 snapshots". Proposed: "Clearing the bank at the start of each trajectory
  constrains the history available to that trajectory's 20 snapshots".
* Fourth paragraph. Current: "together with the similarity weight of the connecting edge".
  Proposed: "together with the two attributes of the connecting edge, its similarity and its
  arrival flag".
* Seventh paragraph. Current: "The sequence of 10 snapshot representations produced by the
  readout". Proposed: "The sequence of snapshot representations produced by the readout".
* Table 3.5 (per component, `R2/table_4_11_profile.json`): node memory 203,904; graph encoder
  340,224; attention pooling 8,321; temporal aggregator 99,072; graph head 8,321; node head
  20,673; total 680,515.

### D6. Section 3.9 and Table 3.6 (training)
* Fifth paragraph. Current: "and a batch size of 4 windows." Proposed: "and a batch size of 4
  sequences." Current: "The second phase was configured for 10 epochs with early stopping on
  validation loss at a patience of 10 epochs, and the checkpoint attaining the lowest validation
  loss was retained for evaluation. The patience was set at the epoch budget deliberately, so that
  the configured schedule would run to completion, with early stopping retained as a safeguard
  against divergence, with no active constraint on training length." Proposed: "The second phase
  was configured for at most 66 epochs, which matches the earlier configuration's 12 epochs over
  windows of 10 snapshots in the number of snapshots processed, with early stopping at a patience
  of 10 epochs on validation area under the curve, and the checkpoint attaining the highest
  validation area under the curve was retained for evaluation." (Apply after A15, which corrects
  the objective weights and the contrastive phase.) Source: FZ2 section 2; IMP step 4.
* Fourth paragraph (augmentation), add at the end: "The arrival-coherence attributes were
  computed from the embeddings before noise was added."
* Table 3.6: epochs 66 (early stopping, patience 10, on validation area under the curve); batch
  4 sequences; window 20 snapshots; arrival window 10 interactions; node attributes 401; edge
  attributes 2; parameters 680,515.

### D7. Section 3.10 (benchmarks, ablations and the combined detector)
* First paragraph. Current: "Five benchmark methods were implemented". Keep, and add at the end
  of the paragraph: "The static graph model and every ablation variant received the
  arrival-coherence attributes and arrival edges, as the detector did."
* Last paragraph, add (after C3's sentence on the three new variants): "A further variant
  removes the ten arrival-coherence attributes and keeps everything else, to measure what they
  contribute. The widened static variant is matched to the detector's 680,515 parameters, with
  680,943."
* New paragraph at the end of the section, proposed: "The configuration reported in Chapter Four
  combines the detector with the static graph model. For each training seed, the sequence scores
  of the detector and of the static model trained under that seed were each converted to ranks
  over the sequences scored and averaged, and the result is reported as the combined detector,
  beside each model alone. Because ranks over a test partition use the scores of the other test
  sequences, a second form replaces each score by its position among the same model's validation
  scores, which a deployment could compute. The combination was chosen by the rule stated in
  Section 3.12 before any test figure for this configuration existed."
* New paragraph after the oracle paragraphs, proposed: "Three further oracles read the
  arrival-coherence attributes. Forty sequence-level aggregates of the ten attributes on the final
  snapshot, being for each attribute its maximum, its mean, the mean of its five largest values
  and its maximum less its median, were supplied to a logistic regression, to gradient boosting
  tuned on the validation partition, and, together with the thirteen features above, to a second
  logistic regression. They establish what the hand-built attributes support without a graph
  model." Source: FZ2 section 5; `oracle_comparison.coherence_aggregates`.

### D8. Section 3.11 (perturbations)
* Fifth paragraph, add at the end: "Because the arrival-coherence attributes and arrival edges
  are derived from the embeddings when a snapshot is built, as the similarity edges are, they are
  computed from the clean embeddings, and each perturbation changes only what it is stated to
  change. Arrival edges are never removed by the threshold sweep or by edge loss, since they
  record order of arrival and not similarity. A fifth sweep applies the stricter reading of
  embedding noise, in which the noisy embeddings also rebuild the similarity edges, the structural
  attributes, the arrival-edge similarities and the arrival-coherence attributes, at the same
  standard deviations." Source: FZ2 Tables 4.7 to 4.10; IMP step 6 amendment.
* Last paragraph: C9's sentence on five seeds and three draws applies. Add: "With one window
  per sequence every robustness figure is a sequence-level figure; the readings over time were
  not repeated under perturbation."

### D9. Section 3.12 (protocol; how the configuration was reached)
* First paragraph, add at the start: "The headline unit is the sequence, scored once from all of
  its snapshots; the eleven readings over time described in Section 3.4 are the window-level
  unit, and Table 4.1 reports both."
* Second paragraph. Current: "At a target of five per cent the detector realised 3.19 per cent
  on test and the nine-feature regression 5.71 per cent; at a target of ten per cent the
  corresponding figures are 8.74 per cent and 11.63 per cent." Proposed: "At a target of five per
  cent the detector realised 3.67 per cent on test and the nine-feature regression 4.48 per cent;
  at a target of ten per cent the corresponding figures are 9.75 per cent and 9.60 per cent."
  (Replaces C4's rates.) Source: FZ2 Table 4.2.
* Fourth paragraph (latency). Current: "Windows were then evaluated in order, beginning with the
  earliest window whose final snapshot falls at or after onset. Detection was recorded at the
  first window whose predicted probability exceeds the model's own calibrated threshold, and the
  detection point was taken as the index of that window’s final snapshot." Proposed: "The
  readings were then evaluated in order, beginning with the earliest reading whose final snapshot
  falls at or after onset. Detection was recorded at the first reading whose predicted
  probability exceeds the model's own calibrated threshold, and the detection point was taken as
  the index of that reading's final snapshot."
* Add after the fourth paragraph, proposed: "One threshold serves every reading in the primary
  analysis. Because the detector is trained on whole trajectories, clean trajectories score
  higher early in a trajectory than late, so a secondary analysis, fixed before any test figure
  was computed, calibrates one threshold for each reading's final snapshot on the benign
  validation readings ending there."
* New paragraph at the end of the section, proposed: "The configuration reported in Chapter Four
  was not the first evaluated. An initial configuration, scoring each sequence as the highest of
  its eleven window scores over similarity edges alone, attained a sequence-level area under the
  curve of 0.6614 on the test partition, and a logistic regression on hand-built statistics of
  consecutive arrivals in the final snapshot attained 0.8298 on the same partition, which showed
  that the input carried more than the detector recovered. Five changes were then examined in
  order, each fixed in writing together with the rule for deciding whether to keep it before it
  was trained, and each judged on the validation partition alone. The evaluation scripts also
  computed and displayed test figures for the intermediate configurations, at window level for
  the second and third and at sequence level for the fourth; none was used to choose anything,
  and the scripts were then changed so that test figures are no longer displayed during
  training. The test partition was therefore
  consulted for the initial configuration before the final one was developed, and the final
  configuration's test figures were computed once, after every run had finished." Sources:
  freeze 1; IMP steps 1 to 6 and the protocol breach note in step 4.

### D10. Section 3.14 (reliability)
* Second paragraph. Current: "The window length was set to 10 snapshots, which was based on a
  typical episodic retrieval span, and was not tuned across alternatives, so the latency results
  are specific to this window length and a shorter or longer window could alter the balance
  between latency and accuracy." Proposed: "The detector reads whole trajectories of 20
  snapshots, so its readings early in a trajectory are taken on inputs shorter than any it was
  trained on, and Section 4.4 reports the consequence for latency."
* Third and fourth paragraphs: C2's changes apply, with the noise floor restated. Proposed for
  C2's Section 3.14 sentence: "Five runs differing only in random seed have a standard deviation
  of 0.0032 at sequence level, and differences are judged by their paired bootstrap intervals."
* Third paragraph, add at the end: "The configuration reported was reached after the initial
  configuration's test figures were known (Section 3.12), which is a further reason the
  validation-to-test comparison of Section 5.3 is reported."

### D11. Section 3.16 and Section 6.2, second paragraph (numbers only)
* "391-dimensional attributed nodes connected by cosine-similarity edges above a threshold of
  0.6" becomes "401-dimensional attributed nodes connected by cosine-similarity edges above a
  threshold of 0.6 and by edges between records that arrived close together".
* "totalling 670,915 parameters" becomes "totalling 680,515 parameters".
* "2,000 sequences and 22,000 windows" and "yielding 22,000 sliding windows" become "2,000
  sequences" and "yielding 2,000 trajectories of 20 snapshots".

### D12. Section 4.1 and Section 4.2
* Section 4.1, second paragraph. Current: "comprising 500 interaction sequences and 5,500 sliding
  windows, of which 1,769 carry a positive label". Proposed: "comprising 500 interaction
  sequences, read at 5,500 points over time, of which 1,769 carry a positive label".
* Section 4.2, replace both paragraphs' figures, proposed first paragraph: "Training followed the
  procedure specified in Section 3.9, under five seeds. Early stopping ended the detector's runs
  after 59, 35, 44, 35 and 60 epochs of the 66 configured, and the checkpoints retained were
  those of epochs 49, 25, 34, 25 and 50, selected by validation area under the curve. The static
  graph model ran to 66 epochs under four of the five seeds." Proposed second paragraph:
  "Validation ranking was assembled gradually. Under seed 1 the validation area under the curve
  stood at 0.70 after the first epoch, 0.85 by the fourth and 0.90 by the twenty-sixth, reaching
  0.9121 at the forty-ninth, and the last epoch stood within 0.001 of the selected one, so the
  choice of checkpoint flatters the result very little." Sources: FZ2 section 2; IMP step 5.
* New paragraph at the end of Section 4.2 (the development record), proposed: "The reported
  configuration was reached as follows, on the validation partition. The initial configuration
  attained 0.6538 across five seeds. Training the window objective only on windows in which an injected record arrives
  attained 0.6931 under seed 1; adding arrival edges attained 0.7049; reading each trajectory as
  a whole attained 0.7206; and adding the arrival-coherence attributes attained 0.9121. Under the
  rule fixed before the last step, the configuration taken forward was the one with the highest
  validation figure, a single model being preferred if it came within 0.01 of the best: the
  combined detector attained 0.9244 against the detector's 0.9121, so the combination was
  taken forward." Source: IMP steps 2 to 5 and the step 5 decision.

### D13. Section 4.3 and Table 4.1 (replaces C1 and C10)
* Table 4.1 (FZ2 Table 4.1), window level becomes "over the eleven readings": graph AUROC
  0.8248 (sequence level 0.9088 [0.8819, 0.9319]); graph AP 0.7698 (0.8936); F1 at 0.5 0.6311;
  FPR at 90% TPR 0.5653; node AUROC 0.8846; node AP 0.5272. Add a row or note for the combined
  detector at sequence level: 0.9221 [0.8975, 0.9424], AP 0.9080.
* First paragraph. Current: "A window is 10 snapshots; a sequence is the whole simulated
  lifetime of one agent." Proposed: "A reading is the detector applied to the snapshots received
  so far; a sequence is the whole simulated lifetime of one agent." Current: "The detector
  attained a sequence-level area under the curve of 0.8122, with a 95% bootstrap interval of
  [0.7707, 0.8532], and an average precision of 0.8185. The corresponding window-level figures
  are 0.8261 and 0.7769." Proposed: "The combined detector attained a sequence-level area under
  the curve of 0.9221 (standard deviation 0.0089 across five seeds), with a 95% bootstrap
  interval of [0.8975, 0.9424], and an average precision of 0.9080. The detector alone attained
  0.9088 (standard deviation 0.0032) with an interval of [0.8819, 0.9319] and an average
  precision of 0.8936, and its figures over the eleven readings are 0.8248 and 0.7698."
  (The position-floor sentences keep C9's 0.5950.)
* Second paragraph. "0.5757" becomes "0.5653"; delete the best-F1 sentence as in C1; "The
  F1-score at the conventional threshold of 0.5 was 0.6691" becomes "0.6311".
* Third paragraph (node level). Current: "A node area under the curve of 0.9087" becomes "A node
  area under the curve of 0.8846"; "A node average precision of 0.5662" becomes "0.5272". Replace
  "Node-level ranking is, notably, the strongest single figure the detector produces, and it
  exceeds its own trajectory-level ranking by a wide margin." with "Node-level ranking, at 0.8846
  over the readings and 0.9043 on the final snapshot, is now below the trajectory-level figure."
  (Replaces C8's sentence for this paragraph; C8's matched-cluster point is in D16.)
* Fourth paragraph (stability): C2's proposed paragraph, with "0.6614 with a standard deviation
  of 0.0144" becoming "0.9088 with a standard deviation of 0.0032", and its last sentence becoming
  "Differences are judged by their paired bootstrap intervals rather than against a fixed
  floor."
* Fifth paragraph (Figures 4.1 and 4.2; replaces C10), proposed figures: seed 5 (the median seed),
  sequence poisoned_placement_0047; "the operating threshold of 0.377" becomes "0.414"; "the
  detector flagged 19 records, recovering 11 of the 12 and attaching 8 benign records to them"
  becomes "the detector flagged 24 records, recovering all 12 and attaching 12 benign records to
  them". The score-range sentence ("concentrated between 0.15 and 0.35 ... 11 sit in a tight group
  near 0.75") must be re-read from `R2/figures/fig_4_1_node_score_distribution.png` before it is
  rewritten; no figure for it is proposed here. Source: FZ2 section 9.

### D14. Section 4.4 and Table 4.2 (replaces C4)
* Table 4.2 (FZ2 Table 4.2, primary): detector realises 3.67% and detects 129.8 of 200 (64.9%),
  median latency 2, mean 2.20, pre-onset alarms 0.100; regression realises 4.48% and detects 64
  (32.0%), median 7, mean 6.73, pre-onset 0.06. Caught within 0, 2, 5 and 10 snapshots: detector
  5.8%, 44.8%, 62.7%, 64.8%; regression 1.5%, 6.0%, 14.0%, 25.5%; differences +0.043 [0.010,
  0.078], +0.388 [0.320, 0.458], +0.487 [0.412, 0.558], +0.393 [0.313, 0.471].
* First paragraph. Current: "At that target the detector realised a false-alarm rate of 3.19 per
  cent on the test partition against the regression's 5.71 per cent". Proposed: "At that target
  the detector realised a false-alarm rate of 3.67 per cent on the test partition against the
  regression's 4.48 per cent".
* Second paragraph, proposed replacement: "The detector detects 130 of the 200 poisoned
  trajectories on average across seeds against the regression's 64, and among those it detects it
  does so sooner, with a median latency of two snapshot intervals against seven. The fraction
  caught within k snapshots of onset, which is defined for every trajectory whether detected or
  not, favours the detector at every horizon by a margin whose interval excludes zero, from 4.3
  percentage points at onset to 48.7 within five snapshots. At a ten per cent target the detector
  detects 150 of 200 against the regression's 119, with a median of one to two intervals against
  seven."
* New paragraph after the second, proposed: "The detector was trained on whole trajectories, and
  read early it scores clean trajectories high: after snapshot 9 the median clean score is 0.352
  and the detector separates poisoned from clean trajectories at 0.633, rising to 0.909 after
  snapshot 19, where the median clean score is 0.005 (Table 4.2b). A single threshold is therefore
  set by the early readings, and the detector raises 0.100 pre-onset false alarms per poisoned
  trajectory at the five per cent target, and 0.361 at ten per cent. With one threshold for each
  reading, fixed before the test partition was scored, it detects 148.4 of 200 at a realised 5.50
  per cent." New Table 4.2b: FZ2 section 7 table. Source: FZ2 section 7 and Table 4.2.
* Third paragraph (Figure 4.3, replaces C10's Figure 4.3 text), proposed: "Figure 4.3 shows the
  detection score across the lifetime of one flood-placement sequence. The score remained below
  0.22 for every reading whose final snapshot preceded the injection, rose to 0.97 at the
  snapshot following a ground-truth onset at snapshot 13, crossed the operating point of 0.9987 at
  snapshot 15, and remained near 1.0 thereafter." Keep the two sentences that follow. Source:
  `R2/figures/fig_4_3_single_sequence_detection_score.json`.

### D15. Section 4.5 and Table 4.3 (replaces C5's table and figures; C5's R1 paragraph stays)
* Table 4.3 (FZ2 Table 4.3, size-matched): 5 to 8 records 0.828; 9 to 12, 0.874; 13 to 16,
  0.925; 17 to 20, 0.973.
* First paragraph, proposed figures: "Attacks of eight records or fewer are detected at 0.828,
  and attacks of seventeen or more at 0.973, with the two intermediate bands between them in
  order."
* Second paragraph. Current: "A campaign small enough to remain below nine records is not
  reliably detected by this model at this configuration, and an adversary aware of the detector
  has an evident incentive to stay there." Proposed: "A campaign below nine records is detected
  less reliably than a larger one, and an adversary aware of the detector has an evident
  incentive to stay small." Delete the node-level sentence ("Node-level ranking degrades more
  gently ... 0.6894 ... 0.9864"), which has no counterpart in freeze 2.
* C5's R1 paragraph: change "and the detector attains 0.6614" to "and the initial configuration
  of the detector attains 0.6614", since R1 concerns that configuration.
* Later mentions (4.11, 5.3, 5.7, 6.2, 6.6) of the size bands become 0.828 and 0.973; sentences
  saying detection is close to chance or weak for small campaigns are replaced by D19 and D20.

### D16. Section 4.6 and Table 4.4 (replaces C6's oracle paragraph and C8's table)
* Table 4.4: unsupervised rows as C8 (they do not read the arrival measures, which the table
  note should state); add "Arrival-coherence attributes alone, logistic regression: 0.8351, AP
  0.7891"; static graph model 0.9156 [0.8916, 0.9367], AP 0.8995, record level 0.9162; detector
  0.9088 [0.8819, 0.9319], AP 0.8936, record level 0.8846; combined detector 0.9221 [0.8975,
  0.9424], AP 0.9080. Source: FZ2 Table 4.4.
* Third paragraph (static model), proposed replacement: "The static graph model produced the
  most consequential comparison, because it differs from the detector in exactly the components
  whose contribution the study sets out to measure. It attains a sequence-level area under the
  curve of 0.9156, with an interval of [0.8916, 0.9367], against the detector's 0.9088, a paired
  difference of -0.0068 with an interval of [-0.0195, 0.0061] that includes zero. Once the
  representation records the order of arrival, the final snapshot carries the signal that the
  initial configuration needed the whole trajectory to approach. The rate at which a cluster
  accumulated still separates the classes, but it is written into the final snapshot by the
  arrival edges and arrival-coherence attributes, so a model reading that snapshot alone recovers
  it. Combining the two models adds 0.0133 over the detector, with an interval of [0.0063,
  0.0206]. Section 4.7 decomposes the difference across components."
* Record-level comparison (C8's proposed sentences), proposed: "Measured against the records of
  the matched benign clusters, which are as dense as an attack, the detector ranks injected
  records at 0.9352 and the static model at 0.9570; the arrival-coherence attributes let a single
  snapshot separate them, where in the initial configuration the static model was at chance
  (0.5461)." Source: FZ2 Table 4.4.
* Oracle comparison (replaces C6's table rows and paragraph). Rows: FZ2 section 5. Proposed
  paragraph: "The detector separates clearly from every model restricted to the final state, from
  every model reading the thirteen hand-built temporal summaries, and from the
  arrival-coherence attributes alone, by 0.074 with an interval of [0.035, 0.111] over the
  logistic regression on their forty aggregates. It does not separate on area under the curve
  from a logistic regression given both the thirteen summaries and the forty aggregates, which
  attains 0.8815 against the detector's 0.9088, a difference of 0.027 with an interval of
  [-0.007, 0.059], while it leads that regression on average precision by 0.052 [0.010, 0.092].
  The graph model therefore adds to the hand-built attributes it is given, and a regression given
  enough hand-built features comes close to it."

### D17. Section 4.7 and Table 4.5 (replaces C3)
* Table 4.5: FZ2 Table 4.5 (eight rows; parameters column included; gated readout row deleted).
* First paragraph, proposed replacement: "Table 4.5 reports the ablation variants specified in
  Section 3.10, each changing one thing and trained under an identical recipe over five seeds.
  Removing the ten arrival-coherence attributes returns the detector to 0.6680, a paired
  difference of 0.2408 with a 95% interval of [0.2036, 0.2767], the level of the initial
  configuration. Of the other removals only one has an interval excluding zero: removing the
  temporal aggregator and keeping the node memory costs 0.0100 with an interval of
  [0.0011, 0.0195]. Removing the node memory and keeping the aggregator leaves the
  detector at 0.9148, removing both leaves the static model at 0.9156, replacing attention pooling
  with mean pooling gives 0.9172, and removing the four structural attributes gives 0.9069, with
  every interval including zero. A static model widened to the detector's size attains 0.9332, a
  difference of -0.0244 with an interval of [-0.0414, -0.0097], so it exceeds the detector."
* Second paragraph (mean pooling): keep the parameter argument (8,321) and replace C3's
  replacement sentence with: "On this configuration the 8,321 parameters of attention pooling buy
  nothing measurable, a difference of -0.0084 with an interval of [-0.0228, 0.0055]."
* Third and fourth paragraphs (the concessions and the structural features), proposed
  replacement for both: "Two things follow, and they reverse the reading of the initial
  configuration. The first is that the temporal machinery is not what detects the attack here: a
  static model of the same size does better, and neither the node memory nor attention pooling can
  be shown to contribute. The second is that the arrival-coherence attributes are what the result
  rests on. They are hand-built from knowledge of how the attack is written, and Section 5.7
  records what that implies. The structural attributes, whose removal left the initial
  configuration at chance, are no longer needed once arrival order is supplied."

### D18. Sections 4.8 to 4.11 and Tables 4.6 to 4.11 (replaces C9 except its Table 3.7 item)
* Table 4.6: probe rows as C9 (they read the data only); add "Arrival-coherence attributes
  (forty aggregates, sequence level) 0.8351 (AP 0.7891)"; detector row "over the eleven readings
  0.8248 (AP 0.7698)".
* Section 4.8, second paragraph. Current (after C9): "leaves a distance of 0.0703 to the
  detector's window-level figure of 0.7044". Proposed: "leaves a distance of 0.1907 to the
  detector's figure of 0.8248 over the readings". Replace the sentence beginning "What separates
  the detector from the structural probe" to the end of the paragraph with: "What separates the
  detector from the structural probe is the arrival-coherence information, which alone supports
  0.8351 at sequence level and whose removal costs the detector 0.2408 (Section 4.7). The
  attribution and the ablation agree, and they locate the signal in the order of arrival rather
  than in the detector's temporal components."
* Tables 4.7 to 4.10: FZ2 Tables 4.7 to 4.10 (sequence level; columns sequence AUROC, AP, F1 at
  0.5, FPR at 90% TPR, node AUROC, node AP); add Table 4.10b (FZ2, extra sweep).
* Section 4.9, first paragraph: the sample sentence becomes "Each sweep was evaluated on all 500
  test sequences, for each of the five trained detectors, with three random draws of every
  stochastic condition." Figures: "Area under the curve fell from 0.9088 without noise to 0.9054
  at a standard deviation of 0.10 and 0.8952 at 0.15, then more steeply to 0.8779 at 0.20, 0.8532
  at 0.25 and 0.8237 at 0.30. Average precision fell from 0.8936 to 0.7885, and node-level
  ranking from 0.9043 to 0.8348."
* Section 4.9, second paragraph. Current: "At a noise standard deviation of 0.30 graph-level
  ranking had fallen to 0.6641 while node-level ranking remained at 0.8363". Proposed: "At a noise
  standard deviation of 0.30 graph-level ranking had fallen to 0.8237 and node-level ranking to
  0.8348, so the two degrade at similar rates on this configuration." Delete the clause from "so
  the detector continues to rank individual records" to the end of the paragraph.
* New paragraph after the second, proposed: "These figures leave the arrival-coherence attributes
  as they were computed from the clean embeddings, as the protocol of Section 3.11 states. When
  the noise reaches them as well, under the fifth sweep, performance collapses: 0.8112 at a
  standard deviation of 0.05, 0.6106 at 0.10 and 0.5444 at 0.15, near chance, and from 0.10
  onward no poisoned sequence scores above 0.5 (Table 4.10b, Figure 4.4b). The attributes depend on precise cosine
  similarities between records, and the detector never encountered noisy attributes in
  training." Source: FZ2 Tables 4.7 to 4.10.
* Threshold sweep paragraphs. Proposed figures: "Graph-level ranking stood at 0.9118 at a
  threshold of 0.4, 0.9115 at 0.5, 0.9088 at the trained value of 0.6 and 0.9040 at 0.8, while
  node-level ranking rose from 0.8781 to 0.9202." Replace C9's "values from 0.5 to 0.6 lie within
  0.003 of each other, and 0.8 costs 0.082" with "every value tested lies within 0.008 of the
  best, with arrival edges kept at every threshold". The next paragraph: delete "both of
  which decline gently and in the same direction beyond a threshold of 0.5" and state that
  graph-level ranking falls slightly and node-level ranking rises as the threshold rises, so the
  two outputs prefer different densities.
* Tables 4.9 and 4.10 paragraph, proposed figures: "Metadata corruption reduced graph-level
  ranking from 0.9088 to 0.9059 at a corruption probability of 0.30, with node-level ranking
  falling from 0.9043 to 0.8787. Random edge loss raised graph-level ranking slightly, to 0.9134 at
  a drop rate of 0.30, and node-level ranking to 0.9197." Add: "Edge loss removes similarity edges
  only, and that removing them does not hurt is consistent with the signal travelling through the
  arrival edges and attributes."
* Summary paragraph of 4.9, proposed: "Under the protocol of Section 3.11 embedding noise is the
  only sensitivity of consequence, costing 0.085 at a standard deviation of 0.30, while metadata
  corruption costs 0.003, threshold variation at most 0.008 and edge loss nothing. When noise
  reaches the arrival-coherence attributes the detector fails, so their integrity is the
  condition on which detection depends." Keep the in-distribution caveat sentence.
* Table 4.11 and Section 4.10 (replaces C9's Table 4.11 item and A23's values): FZ2 Table 4.11.
  "a single window required a mean of 51.75 milliseconds with a 95th percentile of 55.79
  milliseconds, and a batch of 4 windows required a mean of 55.90 milliseconds, yielding
  throughputs of 19.32 and 71.55 windows per second respectively. Peak memory occupancy was
  116.88 megabytes and the trained checkpoint occupied 2.56 megabytes. Processing on the central
  processing unit alone required a mean of 53.70 milliseconds per window at a batch size of 1,
  rising to 134.28 milliseconds at a batch size of 4." becomes "a single trajectory of 20
  snapshots required a mean of 209.13 milliseconds with a 95th percentile of 260.31
  milliseconds, and a batch of 4 required a mean of 259.75 milliseconds, yielding throughputs of
  4.78 and 15.40 trajectories per second respectively. Peak memory occupancy was 36.55 megabytes
  at a batch size of 1 and 104.78 at 4, and the trained checkpoint occupied 2.61 megabytes.
  Processing on the central processing unit alone required a mean of 199.73 milliseconds per
  trajectory at a batch size of 1, rising to 593.01 milliseconds at a batch size of 4. Computing
  the arrival edges and arrival-coherence attributes for one snapshot required a mean of 5.89
  milliseconds on the central processing unit." Delete "Peak memory occupancy was identical at
  both batch sizes because the dominant allocation is the node memory bank". In the next
  paragraph "19.32 to 71.55 windows per second" becomes "4.78 to 15.40 trajectories per second",
  "19.32 windows per second" becomes "4.78 trajectories per second", and "inference on the
  resulting window completes in approximately 52 milliseconds" becomes "inference on the whole
  trajectory completes in approximately 210 milliseconds".
* Section 4.11 (chapter summary): restate with the figures above: combined detector 0.9221
  [0.8975, 0.9424], AP 0.9080; detector 0.9088; static model 0.9156; 64.9 per cent against 32.0
  per cent at the five per cent target; size bands 0.828 to 0.973; arrival-coherence attributes
  removed 0.6680; widened static model 0.9332; probes as C9; collapse under the fifth sweep.
  Proposed second paragraph: "The ablation and attribution analyses locate the source of that
  result in the arrival-coherence attributes, whose removal costs 0.2408 and which alone support
  0.8351, and not in the temporal components, whose removal costs nothing measurable. The probes
  establish that content alone supports 0.4967 and adds nothing when combined with structure.
  Robustness testing identified one condition on which detection depends, the integrity of the
  arrival-coherence attributes under embedding noise. Chapter Five interprets these findings
  against the research questions."

### D19. Chapter Five
* 5.2, first paragraph. Current: "encodes each record as a 391-dimensional node combining a
  semantic embedding with three behavioural attributes and four structural attributes, and
  connects records by weighted undirected edges above a cosine similarity threshold of 0.6."
  Proposed: "encodes each record as a 401-dimensional node combining a semantic embedding with
  three behavioural attributes, four structural attributes and ten arrival-coherence attributes,
  and connects records by weighted undirected edges above a cosine similarity threshold of 0.6
  and by edges between records that arrived within 10 interactions of each other."
* 5.2, second paragraph. Current: "A detector trained on it attained a sequence-level area under
  the curve of 0.8122". Proposed: "A detector trained on it attained a sequence-level area under
  the curve of 0.9088, and 0.9221 combined with the static model". Current: "Graph-level ranking
  varies by less than 0.043 across thresholds from 0.4 to 0.8". Proposed: "Graph-level ranking
  varies by less than 0.008 across thresholds from 0.4 to 0.8".
* 5.2, third paragraph, add at the end: "The representation also carries hand-built
  arrival-coherence attributes, and the evaluation shows that the result depends on them
  (Section 4.7), so the representation as evaluated is specific to an attack that writes its
  records in a consecutive burst."
* 5.3, first paragraph (replaces C7), proposed: "The acceptance criterion of 0.85 stated in
  Section 3.12 was met at sequence level. The detector attains 0.9138 on the validation partition
  and 0.9088 on the test partition, and the combined detector 0.9260 and 0.9221. Over the
  eleven readings the detector attains 0.8387 and 0.8248, below the criterion. Validation and test
  agree to within 0.005, although the configuration was developed on the validation partition over
  five steps, so selection flattered the validation figure very little." Keep the final sentence
  in a revised form: "The figure establishes that the signature of flood placement can be learned
  with high reliability once the representation records the order of arrival, and the ablation
  evidence establishes that what is learned is carried by the arrival-coherence attributes."
  Source: FZ2 sections 3 and 6.
* 5.3, second paragraph: "0.5757" becomes "0.5653"; the size sentence becomes "with detection
  standing at 0.828 for attacks of eight records or fewer and 0.973 for attacks of seventeen or
  more"; "The aggregate figure of 0.8122" becomes "The aggregate figure of 0.9088".
* 5.3, third paragraph, proposed replacement: "On the second part of the question, the
  comparison in Table 4.2 is more informative than any single latency figure. At a matched five
  per cent false-alarm target the detector detected 130 of 200 poisoned trajectories on average
  against 64 for the nine-feature regression, with a median latency of two snapshot intervals
  against seven, and it caught a larger fraction within every horizon examined, by 0.043 at the
  onset snapshot itself and by 0.393 within ten snapshots, with intervals excluding zero
  throughout. That advantage holds although the detector is weakest early in a trajectory, where
  it was not trained (Table 4.2b): after snapshot 9 it separates poisoned from clean trajectories
  at 0.633, and its single threshold is set by the high scores clean trajectories receive there.
  Training on partial trajectories would address this and was not attempted."
* 5.4, third and fourth paragraphs (the static model), proposed replacement: "The static graph
  model produced the most consequential comparison, because it differs from the detector in
  exactly the components whose contribution the study sets out to measure. It attains 0.9156
  against the detector's 0.9088, a paired difference that includes zero, and a static model of
  the detector's size attains 0.9332, above it. The expectation with which the study began, that
  temporal modelling would be responsible for a substantial part of the result, is therefore not
  borne out on the final configuration. What the initial configuration's static model lacked was
  not the capacity to read several snapshots but the order of arrival, which the similarity graph
  did not show: once arrival edges and arrival-coherence attributes write that order into the
  final snapshot, a model reading that snapshot alone recovers the rate at which the cluster
  accumulated. The conceptual framework of Section 2.8 is supported in what it says about the
  signal, the rate of arrival, and not in what it implied about the mechanism needed to read it."
* 5.4, fifth paragraph ("What the temporal components supply"), proposed replacement: "What
  supplies the capacity to discriminate is therefore the arrival-coherence information. The
  evidence is of two kinds and they agree. The ablation removes the ten attributes and costs
  0.2408, with an interval excluding zero, and the attributes alone support 0.8351 in a logistic
  regression. The detector adds to them, leading that regression by 0.074 with an interval
  excluding zero, but a static model given the same attributes does as well as the detector.
  Both measurements rest on 200 poisoned test sequences."
* 5.5, first paragraph, proposed replacement: "The fourth research question had three parts. On
  component contribution, no component of the detector's temporal machinery can be shown to
  contribute on the final configuration: removing the node memory, removing both temporal
  components and replacing attention pooling with mean pooling each leave a difference whose
  interval includes zero, and removing the temporal aggregator alone costs 0.0100. The design
  rationale of Sections 2.6 and 3.8 for attention pooling is not confirmed here. The largest
  single effect belongs to the arrival-coherence attributes, whose removal costs 0.2408."
* 5.5, second paragraph. Current: "The remaining distance from 0.6818 to the detector's
  window-level figure of 0.8261, which is 0.1443, is the capacity to read the same structural
  quantities across ten snapshots rather than at one, and the ablation measures that capacity
  independently at 0.2517." Proposed (with C9's probe figures): "The remaining distance from
  0.6341 to the detector's figure of 0.8248 over the readings, which is 0.1907, is carried by the
  arrival-coherence information, which alone supports 0.8351 and whose removal the ablation
  measures independently at 0.2408."
* 5.5, third paragraph, proposed replacement: "On robustness, the detector tolerates every
  perturbation that leaves the arrival-coherence attributes intact: embedding noise costs 0.085 at
  a standard deviation of 0.30, and metadata corruption, threshold variation and edge loss cost
  less than 0.01. When the noise reaches the attributes, detection is near chance from a standard
  deviation of 0.15. A deployment would therefore need to monitor the quality of the embeddings
  from which the attributes are computed, since that is the single input whose degradation the
  detector does not survive."
* 5.5, fourth paragraph (node level more robust): delete; on the final configuration the two
  outputs degrade at similar rates (FZ2 Tables 4.7 to 4.10).
* 5.5, fifth paragraph (oracle; replaces C6's Section 5.5 item), proposed replacement: "One
  further comparison belongs here because it is the sharpest test the study applies to itself.
  The thirteen-feature regression of Section 3.10 and the forty arrival-coherence aggregates are
  oracles, hand-built from knowledge of how the attack is generated. Given both, a logistic
  regression attains 0.8815, and the detector's lead of 0.027 on area under the curve has an
  interval of [-0.007, 0.059], while its lead of 0.052 on average precision, [0.010, 0.092],
  excludes zero. The honest statement is that the detector ties a sufficiently informed oracle on
  whole-ranking quality and leads it at the top of the ranking."
* 5.6, first paragraph. Current: "and the ablation evidence establishes that its performance
  depends on temporal information, costing 0.2517 when that information is removed." Proposed:
  "and the ablation evidence establishes that its performance depends on information about the
  order of arrival, costing 0.2408 when the arrival-coherence attributes are removed, and not on
  the temporal components of the model." "a partitioning protocol that prevents leakage between
  overlapping windows" becomes "a partitioning protocol that prevents leakage between overlapping
  readings".
* 5.6, second paragraph. Current: "or that it would detect campaigns small enough to remain below
  nine injected records, where Section 4.5 reports performance close to chance." Proposed: "or
  that it would detect an attacker who spreads the injection out instead of writing it in a
  burst, which the arrival-coherence attributes are not built to see."
* 5.6, third paragraph. Current: "so the whole of the detectable signal is relational and
  temporal." Proposed: "so the whole of the detectable signal lies in how records relate to one
  another and in the order in which they arrived." Delete "since detection tracks the quantity and
  the concentration of injected material," and keep the rest.
* 5.7, first paragraph: C2's replacement applies, with "0.0144" becoming "0.0032" and the noise
  floor sentence replaced by "differences are judged by their paired bootstrap intervals".
* 5.7, second paragraph (attack size), proposed replacement: "The second limitation is the
  dependence on hand-built arrival-coherence attributes, and it is the one bearing most heavily on
  any operational claim. Without them the detector returns to 0.6680. They measure how alike the
  records of a consecutive run are, which is how this attack is written; an adversary who
  interleaves injected records with ordinary traffic, as the matched clean clusters arrive, would
  not produce the runs they detect, and no benchmark of that kind was evaluated. The attributes
  also fail under noise in the embeddings from which they are computed, falling to near chance at
  a standard deviation of 0.15. Detection is weaker for small campaigns, at 0.828 for eight
  records or fewer against 0.973 for seventeen or more."
* 5.7, fourth paragraph. Current: "The 384 embedding dimensions contribute nothing, since removing
  the four structural node columns leaves the model at chance". Proposed: "The 384 embedding
  dimensions contribute nothing directly; the signal reaches the model through the similarities
  between them, in the arrival edges and arrival-coherence attributes". Current: "so exceeding them
  on average precision is meaningful precisely because they encode what the attack is". Proposed:
  "and the arrival-coherence attributes are of the same kind, since they too encode what the
  attack is". Current: "The fourth is that the window length was fixed at 10 snapshots without
  evaluation of alternatives, so the latency results are specific to that choice." Proposed: "The
  fourth is that the detector was trained on whole trajectories only, so it is weakest early in a
  trajectory, where Table 4.2b reports a separation of 0.633 after snapshot 9."
* 5.7, new paragraph at the end, proposed: "A procedural limitation is also recorded. The final
  configuration was developed after the initial configuration's test figures were known, over
  five steps judged on the validation partition, and test figures for intermediate configurations
  were displayed during that development (Section 3.12). No choice was made on a test figure, the rules were fixed in
  writing before each step, and the validation and test figures agree to within 0.005, but the
  test partition is not untouched in the strict sense."
* 5.8, first paragraph. Current: "A model restricted to the final snapshot attains 0.5602 where
  one reading the same quantities across ten snapshots attains 0.8122, so what matters is not the
  state of the store but the rate at which it changed." Proposed: "A model restricted to the final
  snapshot attains 0.9156 once that snapshot records the order of arrival, so what matters is the
  rate at which the store changed, and that rate can be read from the order in which its records
  arrived."
* 5.8, second paragraph. "approximately 52 milliseconds" becomes "approximately 210 milliseconds
  per trajectory"; "a false positive rate of 0.5757" becomes "0.5653".
* 5.8, third paragraph (fifth implication; node level stronger and more robust), proposed
  replacement: "A fifth implication follows from the dependence on the arrival-coherence
  attributes. A deployment relying on them should monitor the quality of the embeddings from which
  they are computed, since the detector does not survive noise that reaches them, and should
  expect an informed adversary to spread an injection out rather than write it in a burst."
* 5.9, both paragraphs: restate with D19's figures and claims (0.9088 and 0.9221; criterion met
  on both partitions; static model 0.9156 and widened 0.9332; arrival-coherence attributes 0.2408
  and 0.8351; the limitations of 5.7).

### D20. Chapter Six
* 6.2, third paragraph: restate with D18's figures; replace "Performance depended strongly on
  attack size, standing close to chance below nine injected records and at 0.9554 above sixteen"
  with "Performance rose with attack size, from 0.828 below nine injected records to 0.973 above
  sixteen"; replace "and the static graph model attained 0.5602, a paired difference of 0.2517
  from the detector whose interval excludes zero" with "the static graph model attained 0.9156,
  not distinguishable from the detector, and removing the arrival-coherence attributes returned
  the detector to 0.6680"; "so that the whole of the detectable signal is relational and temporal"
  becomes "so that the whole of the detectable signal lies in relations between records and in
  their order of arrival".
* 6.3, third paragraph (third and fourth contributions), proposed replacement: "The third
  contribution is the detection approach: a representation that records the order of arrival
  through arrival edges and arrival-coherence attributes, read by a graph model over the whole
  trajectory. Combined with a static graph model it attained a sequence-level area under the curve
  of 0.9221 and an average precision of 0.9080 on the held-out partition, and the detector alone,
  comprising 680,515 parameters, detected 64.9 per cent of poisoned trajectories at a matched five
  per cent false-alarm target against 32.0 per cent for the strongest feature-based baseline. Its
  computational profile, at approximately 210 milliseconds per trajectory on consumer-class
  hardware, establishes that performance of this kind does not require datacentre resources. The
  fourth is a comparative evaluation against the implemented benchmark methods under identical
  partitions, which established that injected records are not detectable as point anomalies, and
  that once arrival order is represented a static graph model matches the temporal detector, so
  that the order of arrival, and not temporal modelling, carries the signal." (Keep A25's count if
  applied.)
* 6.4, first paragraph: "approximately 52 milliseconds" becomes "approximately 210 milliseconds
  per trajectory".
* 6.4, second paragraph, add after "Three operational cautions follow from the results.":
  "Detection depends on hand-built attributes of the attack's arrival pattern, so a deployment
  should expect an informed adversary to change that pattern." and change "Three" to "Four".
* 6.4, third paragraph: "A node average precision of 0.5662" becomes "0.5272"; "0.5757" becomes
  "0.5653"; delete "that record-level ranking is both the stronger and the more robust of the two
  and"; "in which 19 records were flagged against 12 genuinely injected" becomes "in which 24
  records were flagged against 12 genuinely injected".
* 6.5, first paragraph: C2 and C3 deletions apply. Replace "The second is a sweep along attack size
  alone, which would locate the transition between the near-chance regime below nine injected
  records and the reliable regime above sixteen" with "The second is a corpus in which the attacker
  interleaves injected records with ordinary traffic, which would test the dependence on
  arrival-coherence attributes recorded in Section 5.7". Add a sixth immediate direction:
  "training on partial trajectories of random length, which would remove the weakness early in a
  trajectory reported in Section 4.4."
* 6.5, second paragraph: replace "Evaluation across window lengths would establish the trade-off
  between latency and accuracy that the fixed choice of 10 snapshots conceals." with "Training
  with noise applied to the arrival-coherence attributes, or learning arrival features instead of
  building them by hand, would test whether the collapse under embedding noise reported in Section
  4.9 can be avoided."
* 6.5, third paragraph: delete the sentences from "The second concerns a component question" to
  the end (the ablation was run, Table 4.5).
* 6.6, first paragraph. Replace "On the held-out partition of 500 sequences the detector attained
  a sequence-level area under the curve of 0.8122 and an average precision of 0.8185, ranked
  compromised records at 0.9087, and at a matched five per cent false-alarm target detected 135 of
  200 poisoned trajectories against 60 for the strongest feature-based baseline" with "On the
  held-out partition of 500 sequences the detector, combined with a static graph model, attained a
  sequence-level area under the curve of 0.9221 and an average precision of 0.9080, and at a
  matched five per cent false-alarm target the detector detected 130 of 200 poisoned trajectories
  against 64 for the strongest feature-based baseline".
* 6.6, second paragraph. Current: "The whole of the detectable signal is therefore relational and
  temporal". Proposed: "The whole of the detectable signal therefore lies in how records relate to
  one another and in the order in which they arrived".
* 6.6, third paragraph, proposed replacement: "The comparison against a static graph model locates
  the contribution of the temporal components, and it is smaller than the study expected. Once the
  representation records the order of arrival, a model restricted to the final snapshot attains
  0.9156 against the detector's 0.9088, and removing the arrival-coherence attributes returns the
  detector to 0.6680. The latency comparison still favours the detector, which caught a larger
  fraction of poisoned trajectories within every horizon examined than a regression given nine
  hand-built features of the final snapshot, by 0.043 at the onset snapshot and 0.393 within ten.
  The mechanism the conceptual framework proposed is the right one in what it identifies: what
  distinguishes a compromised store is the rate at which it changed. That rate is legible in the
  order in which records arrived, and a single snapshot that records that order is enough to read
  it."
* 6.6, fourth paragraph. Current: "Detection is close to chance for campaigns of eight injected
  records or fewer, and an adversary aware of the detector has an evident reason to stay there."
  Proposed: "Detection rests on hand-built attributes of a burst, fails when noise reaches them,
  and is weaker for campaigns of eight injected records or fewer; an adversary aware of the
  detector has an evident reason to spread an injection out." Current: "the study establishes that
  temporal graph-based detection of memory poisoning is feasible". Proposed: "the study establishes
  that graph-based detection of memory poisoning from the order in which records arrive is
  feasible".

### D21. Figures
* Figures 4.1 to 4.3 (replaces C10): `experiments/rebuild_coherence/figures/` (seed 5, the median
  seed; sequence poisoned_placement_0047, chosen by the same rule). The missed example,
  poisoned_placement_0463, is flagged at record level (all 12 injected records, 8 benign) but
  raises no sequence alert, because its highest reading, 0.9982, is below the threshold of 0.9987
  set by the early clean readings; it can illustrate the latency limitation in Section 5.7
  (`experiments/rebuild_coherence/figures_missed_example/`).
* Figures 4.4 to 4.7 (replaces C11): `experiments/rebuild_coherence/figures/fig_4_4` to
  `fig_4_7`, captions as C11 but "full test partition of 500 sequences"; the window lines coincide
  with the sequence lines and are not drawn. New Figure 4.4b
  (`fig_4_4b_embedding_noise_full.png`), caption: "Detection performance when embedding noise also
  reaches the arrival edges and arrival-coherence attributes (mean and standard deviation over
  five training seeds and three random draws)".

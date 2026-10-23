# Thesis change list (brief v2, Section 2, plus audit findings)

For: Moses, applying edits to `MosesWinboodAyandauTGNN-3.docx` (13 Oct 2026).
Evidence for every factual change is in `AUDIT.md` in this folder; section numbers there are
given as "Audit 1.2" and so on.

Conventions: each item gives location, current text, proposed text and reason. Proposed text
keeps the surrounding sentence and changes as little as possible. British English, no em or en
dashes in prose, no equations.

**Updated 23 October 2026:** group C now follows `RESULTS_FREEZE.md` (main corpus
ap_burst2k_nodup). Groups A and B are unchanged except A13 and A21b, which point to C.

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

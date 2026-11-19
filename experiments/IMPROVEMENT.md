# Improving the detector beyond the results freeze (from 27 October 2026)

The freeze reports sequence AUROC 0.6614 (SD 0.0144, five seeds) on `ap_burst2k_nodup`. This
file records every attempt to improve on it, in order, including the ones that do not help, so
that whatever is finally reported can be traced to a configuration chosen in advance of seeing
test results.

## Protocol

* A configuration is judged on **validation** sequence AUROC only (300 sequences), against the
  freeze's own validation figures: detector seed 1 0.6590, mean of five seeds 0.6538; static
  GAT seed 1 0.5680, mean of five seeds 0.5578.
* A pilot is one seed. A configuration that helps on validation gets five seeds; only then is
  its test AUROC read, once.
* Every change made to the detector is also made to the static GAT, so the temporal-versus-
  static comparison stays like for like.
* Nothing in the freeze is overwritten. New runs live in their own folders.

## Step 1. Ceiling check (done)

`python -m src.experiments.ceiling_check`; output `experiments/ceiling/ceiling_check.json`.
Each oracle fitted on train, selected on validation, scored once on test (500 sequences).

| Model | Sequence AUROC [95% CI] | Size 5-8 | 9-12 | 13-16 | 17-20 |
|---|---|---|---|---|---|
| Detector, five seeds | 0.6614 | 0.618 | 0.608 | 0.649 | 0.761 |
| A. Arrival-run scan, embeddings and arrival order only, LR | 0.8298 [0.793, 0.866] | 0.713 | 0.829 | 0.908 | 0.863 |
| A. Same features, gradient boosting | 0.7964 | 0.684 | 0.794 | 0.846 | 0.849 |
| B. Generator oracle (knows how attacks are drawn) | 0.9999 | 0.999 | 1.000 | 1.000 | 1.000 |

Size columns: poisoned sequences against clean controls carrying a cluster of the same size.

What it shows:

* The benchmark is not information-limited: with knowledge of the generator the attack is
  detected perfectly (B).
* The detector's own input supports at least 0.83 (A uses only what the detector receives,
  from a single final snapshot). The detector extracts much less than is there.
* Why: similarity edges exist only above cosine 0.6, and only 25% of the pairs inside an attack
  burst clear it (mean pairwise cosine 0.43, against 0.20 for an ordinary run of records). The
  graph shows three quarters of a burst as unconnected.
* Consequence for the freeze: the "tie with the thirteen engineered features" (0.6864) is a tie
  with weak features. A stronger engineered scan reaches 0.83. This must be stated in the thesis
  whatever the outcome of the steps below.

## Step 2. Window target "arrival" (pilot done)

Positive windows in which no injected record arrives (the injection was complete before the
window began, and memory is reset per window) are left out of the window loss and of the
validation AUROC used to pick the checkpoint. 697 of 13,200 training windows (17% of positives).
Test scoring is unchanged. Code: `dataset.window_is_stale`, `--window-target arrival`.

| Run | Validation sequence AUROC | AP | Status |
|---|---|---|---|
| detector seed 1 | 0.6931 | 0.6539 | done 27 Oct 17:16, 3.0 h training, `experiments/option2_arrival/tgn_s1/` |
| freeze detector seed 1 (same seed, presence target) | 0.6590 | 0.6288 | reference |
| freeze detector, five seeds | 0.6384 to 0.6759, mean 0.6538 | | reference |

Paired difference from freeze seed 1 on the same 300 validation sequences: +0.0341, 95%
bootstrap interval [-0.0116, +0.0769] (2,000 resamples). Above every freeze seed, but the
interval includes zero: a modest gain, not yet established. The masked window AUROC in the
training log (0.7256 at epoch 12) is not comparable with the freeze's 0.7061, because it leaves
out the hardest positives.

`train_variant` writes the test window AUROC to `variant.json` after scoring, as it did for every
freeze run (0.7038 here); it was not used for anything. The test sequence AUROC has not been
computed and will not be until a configuration is fixed.

Not yet done for the static GAT. Before a five-seed comparison, `static_attn` gets the same
target so the comparison stays like for like.

## Step 3. Arrival edges (pilot running since 27 Oct 17:16)

Every two records that arrived at most 10 interactions apart are joined by an edge carrying
their cosine and a flag; similarity edges keep flag 0. W = 10 (one snapshot interval) was fixed
before any run. Structural node features are unchanged. Code: `dataset.with_arrival_edges`,
`--arrival-window 10`. Adds 640 parameters to each model (1,190 to the wider static model).
On arrival edges, attack-to-attack pairs have mean cosine 0.69 against 0.16 for benign pairs.

| Run | Validation sequence AUROC | Status |
|---|---|---|
| detector seed 1 | 0.7049 (AP 0.7021) | done 27 Oct 22:38, 4.8 h training (671,555 parameters), `experiments/option_arrival_edges/tgn_s1/` |
| static GAT seed 1 | 0.6977 (AP 0.6650) | done 28 Oct 01:22, 2.6 h training (306,883 parameters), `experiments/option_arrival_edges/static_attn_s1/` |

Detector: paired difference from freeze seed 1 on the same 300 validation sequences +0.0459,
95% bootstrap interval [-0.0129, +0.1017]. Above every freeze seed, interval includes zero.
Validation window AUROC 0.7567 against 0.7061 for freeze seed 1: the gain is larger per window
than per sequence, so the window-max aggregation over a sequence is losing part of it. Test
window AUROC written by `train_variant`: 0.7287 (not used).

Static GAT: paired difference from freeze static seed 1 +0.1297 [+0.0694, +0.1947]. Validation
window AUROC 0.7359 (freeze 0.6298); node AUROC 0.885, above the detector's 0.849.

**Detector against static GAT, same seed, validation sequences:**

| | Detector | Static GAT | Detector minus static [95% CI] | Rank correlation |
|---|---|---|---|---|
| Freeze (similarity edges only) | 0.6590 | 0.5680 | +0.0910 [+0.0266, +0.1558] | 0.550 |
| With arrival edges | 0.7049 | 0.6977 | +0.0072 [-0.0363, +0.0495] | 0.788 |

The edges lift the static model five times as much as the detector, and the temporal advantage
of the freeze all but disappears. Reading: most of what the detector gained over the static
model came from recovering arrival order, which the similarity graph did not show. Once the
graph shows it, a single snapshot carries most of the signal. This has to be stated in the
thesis whatever step 4 finds: the temporal-versus-static claim, as frozen, depends on a graph
that hid arrival order. Test window AUROC written by `train_variant` for the static run: 0.7024
(not used).

This pilot keeps the reported "presence" target, so it measures the edges alone. If both
steps help, the combination needs its own validation pilot before any five-seed run.

## Step 4. Memory over the whole sequence (option 3)

Decided 28 Oct 01:25, after steps 2 and 3, on validation only. Reasons: the best sequence
AUROC so far (0.7049) is short of the target; per window the detector already reaches 0.7567,
so the max over eleven windows, each starting from wiped memory, loses signal; and with arrival
edges the static GAT nearly matches the detector, so the one thing only the detector can do,
carry memory, has not yet been given the whole sequence to work with.

Fixed before the run:

* `--temporal-window 20`: one window per sequence, covering all 20 snapshots, memory reset once
  per sequence. The window score is the sequence score, so the max aggregation goes.
* Arrival edges W = 10 kept (step 3). Window target "presence": with one window starting at the
  first snapshot, a window is stale only if the whole injection falls in the first ten
  interactions, so step 2's mask has nothing to act on.
* 66 epochs: one window per sequence gives 1,200 training windows of 20 snapshots instead of
  13,200 of 10, and 66 epochs matches the freeze's 12 epochs in snapshot passes
  (12 x 13,200 x 10 = 1,584,000 = 66 x 1,200 x 20). Everything else in the recipe unchanged,
  including the cosine schedule (over 66 epochs) and early stopping patience 10.
* Same for the static GAT, which then sees the final snapshot of each sequence.

Caveat, stated in advance: the checkpoint is chosen on validation AUROC, which at one window per
sequence is the validation sequence AUROC itself, so the selected figure is biased upwards
relative to steps 2 and 3 (chosen on window AUROC). The last epoch's validation AUROC
(`training_history.json`) is reported beside it as the unselected figure.

| Run | Validation sequence AUROC (selected epoch) | Last epoch | Status |
|---|---|---|---|
| detector seed 1 | 0.7206 (epoch 9) | 0.7137 (epoch 19) | early-stopped at epoch 19 of 66, 1.2 h, `experiments/option3_sequence/tgn_s1/` |
| static GAT seed 1 | 0.6600 (epoch 13) | 0.6352 (epoch 23) | early-stopped at epoch 23 of 66, 0.8 h, `experiments/option3_sequence/static_attn_s1/` |

Detector: paired difference from the arrival-edge detector of step 3 (window max), +0.0157
[-0.0372, +0.0700]; from freeze seed 1, +0.0616 [-0.0049, +0.1290]. Sequence scores of step 3
and step 4 have rank correlation 0.672.

Early stopping (patience 10, as fixed in advance) fired at epoch 19 of 66, ten epochs after a
single high epoch (0.7206 at epoch 9, the next best 0.7171 at epoch 17), with the cosine
schedule still at about 80% of its starting learning rate. The 66-epoch budget meant to match
the freeze's compute was therefore not used; 19 epochs is 29% of it.

**Protocol breach, recorded as it happened.** At one window per sequence the window AUROC
`train_variant` writes for the test split *is* the test sequence AUROC, and the queue printed it
to `queue.log`, which was read at 02:48 to see whether the run had finished: 0.6551. This was
not anticipated when step 4 was fixed. Consequences and remedies:

* Nothing about step 4 is decided on that figure. Whether option 3 is pursued is argued from the
  validation figures above alone, and this breach is to be disclosed wherever step 4 is reported.
* From now on `train_variant --score-splits val` (passed through `seed_queue`) scores validation
  only for pilots, and `seed_queue` prints and tabulates validation AUROC, never test.
* The static GAT run of this step was already running under the old code, so its test score will
  be computed and written to its `variant.json` and `queue.log`. Neither will be read: its
  validation figures are taken from `window_scores_val_b1.npz` and the `val_` fields only.
  (Done: read 28 Oct 03:36 with the `test` fields filtered out.)

Static GAT at one window per sequence (it then reads only the final snapshot of each sequence,
1,200 training graphs): 0.6600, below its step 3 figure of 0.6977. Detector minus static at
step 4, same seed: +0.0606 [-0.0285, +0.1545]; detector of step 4 minus the best static
configuration so far (step 3): +0.0229 [-0.0293, +0.0747].

### Validation summary after steps 1 to 4 (seed 1 throughout)

| Configuration | Detector | Static GAT |
|---|---|---|
| Freeze | 0.6590 | 0.5680 |
| Step 2, window target | 0.6931 | not run |
| Step 3, arrival edges | 0.7049 | 0.6977 |
| Step 4, arrival edges and one window per sequence | 0.7206 (0.7137 last epoch) | 0.6600 (0.6352 last epoch) |

No single configuration reaches 0.75 on validation, and no single step is outside its own
bootstrap interval.

### Exploratory, validation only: rank-averaged pairs (28 Oct 03:40)

Prompted by the step 4 sequence scores of detector and static GAT having rank correlation 0.036.
All six pairs of the four step 3 and step 4 runs were examined; every one is listed, because
picking the best of six on 300 sequences flatters the winner. Seed ensembles of the freeze are
given as the control for "any two models help".

| Rank average of | Validation sequence AUROC | Rank correlation of the pair |
|---|---|---|
| Freeze detector, five seeds | 0.6638 (members 0.638 to 0.676) | |
| Freeze static GAT, five seeds | 0.5590 (members 0.545 to 0.568) | |
| Step 4 detector + step 4 static | **0.7581** | 0.036 |
| Step 3 detector + step 4 static | 0.7459 | -0.029 |
| Step 3 static + step 4 static | 0.7419 | -0.055 |
| Step 4 detector + step 3 static | 0.7251 | 0.678 |
| Step 4 detector + step 3 detector | 0.7243 | 0.672 |
| Step 3 detector + step 3 static | 0.7109 | 0.788 |

By cluster size (poisoned against clean controls of the same size), step 4 detector + step 4
static: 5-8 0.659, 9-12 0.727, 13-16 0.781, 17-20 0.812; above both members in every band.

Reading: the gain comes from the step 4 static model ranking sequences almost independently of
every other model, not from pairing a temporal with a static model: two static models together
reach 0.7419. Seed ensembles add only about 0.01. What the step 4 static model picks up has not
been established. This is a lead found after looking, not a result, and it is not to be reported
as one unless confirmed by a configuration fixed in advance, run over five seeds, and read once on
test.

## Step 5. Arrival-coherence node features (pilot; chosen by Moses 28 Oct, written before the run)

Ten columns appended to every record's features, computed per snapshot from the embeddings and
arrival order only (`dataset.with_coherence_features`, `--coherence-features`): for block
lengths 5, 10, 15 and 20, the highest mean pairwise cosine over the runs of consecutive records
containing the record, and that minus its median over the snapshot; and the mean cosine to the
records that arrived within 5 and within 10 interactions. They are the per-record form of the
step 1 scan. Checked against a brute-force implementation (agreement to 5e-8). On 400 *training*
sequences, single columns separate poisoned from clean at 0.66 to 0.74 (max over records of the
final snapshot).

What this changes about the claim: part of any gain is then owed to hand-built features, and the
thesis must say so. The static GAT receives them too.

Configuration: step 4 plus the features (one window per sequence, arrival edges W = 10, 66
epochs, patience 10, presence target), detector and static GAT, seed 1, **validation scored
only** (`--score-splits val`). Step 4 is the base because it gave the best single detector and
lets the static GAT read the whole final snapshot, the setting in which the step 1 scan was
measured.

**Decision rule, fixed now.** The configuration taken to five seeds is the one with the highest
validation sequence AUROC among: (A) step 4 detector + step 4 static, rank mean, 0.7581; (B) step 5
detector alone; (C) step 5 detector + step 5 static, rank mean. A candidate within 0.01 of the
best is preferred if it is a single model. A's figure is the best of six pairs and flattered;
that is accepted, not corrected. Whatever is chosen, its static counterpart (the static GAT of the
same step, alone) is run over the same five seeds and reported beside it, and the five-seed
figures of the freeze stay the reference. If the five-seed runs come to more than 12 GPU-h,
Moses is asked first.

| Run | Validation sequence AUROC (selected epoch) | Last epoch | Status |
|---|---|---|---|
| detector seed 1 | 0.9121 (epoch 49) | 0.9112 (epoch 59) | early-stopped at epoch 59 of 66, 6.2 h (680,515 parameters), `experiments/option5_coherence/tgn_s1/` |
| static GAT seed 1 | 0.9059 (epoch 59) | 0.9054 (epoch 66) | all 66 epochs, 4.5 h (312,003 parameters), `experiments/option5_coherence/static_attn_s1/` |

**Decision (28 Oct 19:25, validation only, applying the rule above as written).**

| Candidate | Validation sequence AUROC | AP |
|---|---|---|
| (A) step 4 detector + step 4 static, rank mean | 0.7581 | |
| (B) step 5 detector alone | 0.9121 | 0.8960 |
| (C) step 5 detector + step 5 static, rank mean | **0.9244** | 0.9103 |
| step 5 static alone, for reference | 0.9059 | 0.8881 |

C minus B: +0.0122, bootstrap 95% interval [-0.0035, +0.0292] (2,000 resamples of the 300
sequences, seed 0). The rank correlation of the detector and static sequence scores is 0.843,
against 0.036 in step 4. B is 0.0122 below C, outside the 0.01 margin within which a single
model is preferred, so **C is the configuration taken to five seeds**. The margin is narrow and
its interval includes zero; that is reported with it, and the rule is not adjusted after seeing
it. The five-seed runs are unchanged, since C is built from the detector and static GAT that
phase A trains anyway. The static GAT nearly matching the detector once both have the features
is itself a result: on this configuration the temporal model adds little over a static one.

Detector: validation node AUROC about 0.92. The curve rises steadily (0.70 after one epoch,
0.85 by epoch 4, 0.90 by epoch 26) and the last epoch is within 0.001 of the selected one, so
checkpoint selection flatters it very little. No test figure exists.

Checks made because the jump is large (28 Oct, validation only):

* `with_coherence_features` reads `data.x[:, :384]` only; it never sees node or graph labels.
* The same features **without any GNN** (40 sequence-level aggregates of the ten columns on the
  final snapshot: max, mean, mean of the top five, max minus median; logistic regression fitted
  on train): **0.8567** on validation. Gradient boosting, grid chosen on validation: 0.8284.
* For reference, the step 1 scan features on validation: logistic regression 0.7793, gradient
  boosting 0.7467 (their test figures were 0.8298 and 0.7964).

So the features carry most of the gain (step 4's 0.72 to 0.86 without a GNN), and the detector
adds about 0.055 on top. Both parts have to be reported: a thesis or paper claiming 0.91 for
the GNN without the features-only figure beside it would overstate what the GNN contributes.
The features are tailored to this attack (a consecutive burst of near-identical records), and
an attacker who spreads the injection out, as the clean controls do, is what they cannot see;
that is a stated limitation, not a footnote. `features_only_val.npz` holds the aggregates.

Speed: the features cost about 2.4 minutes per epoch (6.3 against 3.9 min). From now on
`CachedTrajectoryDataset` computes them once per snapshot and keeps them (checked bit-identical
to computing them every time, windows of 20 and of 10, first and repeated reads). This affects
runs started after 28 Oct 15:00 only; the static run above started before it.

## Step 6. The full five-seed process on the step 5 configuration (written 28 Oct 16:10, before any run)

Moses, 28 Oct: "I want the full process like we did the last to make everything consistent."
So everything the freeze (`experiments/audit/RESULTS_FREEZE.md`) reports is rebuilt on the step 5
configuration, in the freeze's order and with its scripts, and written to a second freeze
document. Everything below is fixed now; no test figure exists for this configuration, and no
choice after this point is made in response to one.

**Configuration, all runs.** One window of 20 snapshots per sequence, arrival edges W = 10,
coherence features, presence target, per-window memory, 66 epochs with patience 10, checkpoint
by validation AUROC, seeds 1 to 5. Seed 1 of the detector and static GAT are the step 5 pilots,
copied from `experiments/option5_coherence/` and scored on test with the others. Root
`experiments/seeds_coherence/` (S2); rebuild outputs `experiments/rebuild_coherence/` (R2).

| Phase | What | GPU estimate |
|---|---|---|
| A | Detector and static GAT, seeds 2 to 5; seed 1 scored on test | about 21 h |
| B | Online scores (below), static-GAT and record-level node metrics, probes, Figures 4.1 to 4.3, robustness sweeps (five seeds, three draws), Figures 4.4 to 4.7 | about 18 h |
| C | Ablations x 5 seeds: memory only, aggregator only, widened static, mean pooling, no structural features, and **no coherence features** (added: it is the ablation that measures what the features contribute) | 60 to 100 h |
| D | Latency, summaries (each run twice and compared byte for byte), oracle comparison, features-only baseline on test, inference profile, cross-scoring on the duplicates benchmark | about 3 h |

The detector and static headline (Tables 4.1, 4.2, 4.4) is summarised once A and B finish;
C and D are fixed here, so reading it then changes nothing that follows.

**Amendment (28 Oct 19:30, after the step 5 decision, before any test figure).** The step 5
rule chose C, so the headline gains the ensemble, defined now:
* per seed pair: detector seed s with static GAT seed s, each converted to ranks over the
  sequences of the split being scored, and averaged. Reported like every other row: mean and
  SD over the five pairs, and the freeze's bootstrap (`duplicates_study.group_stats`, 2,000
  resamples, seed 0) for the interval. The rank mean uses no labels; it does use the other
  test sequences' scores, which a deployed detector would not have;
* so a secondary, deployable form is fixed with it: each score replaced by its position among
  the same model's validation scores, then averaged;
* reported beside it, on the same pairs and resamples: the detector alone (B), the static GAT
  alone, and the paired differences C minus B, C minus static, B minus static;
* output `experiments/seeds_coherence/ensemble_headline.json` (`coherence_plan ensemble`, run
  twice in D1 and compared byte for byte).

C is reported at sequence level only. The online level, latency, sweeps, figures, ablations,
oracle comparison and duplicates cross-scoring stay the detector's, as fixed above: they
describe the temporal model, and an ensemble of rank-normalised scores has no online or
perturbed counterpart without new definitions. That limitation is stated with the headline.
Checked on a mock in which validation scores stand in for test: with test equal to validation,
the deployable form gives the same AUROC as the primary.

**Definitions this configuration forces, fixed now:**

* *Sequence score* = the score of the one 20-snapshot window (pre-registered in step 4). Not the
  maximum over online scores.
* *Window level* (Table 4.1, Table 4.6's detector row, the validation-against-test check) becomes
  *online* level: the trained model run on snapshots 0 to e, for e = 9 to 19. These are the freeze
  windows' 11 end points with the same labels (5,500 on test), so the population is unchanged.
  Node scores are taken at each end point's snapshot, as before.
* *Latency* (Table 4.2, Figure 4.3) uses the online scores, threshold calibrated on benign
  validation online scores. The nine-feature regression's scores and thresholds are reused from
  the freeze (`experiments/seeds_nodup/latency_*`): same data and windows, and they do not depend
  on the detector.
* *Sweeps* (Tables 4.7 to 4.10), **amended 28 Oct 16:15, before any sweep ran**: the freeze's
  rule is kept. Everything derived from the embeddings when a snapshot is built (similarity
  edges and weights, structural columns, and now arrival edges and coherence columns) comes from
  the clean embeddings, and each perturbation changes only what it changed in the freeze: noise
  the 384 embedding columns, threshold and edge loss the similarity edges (arrival edges are
  never dropped; they encode arrival order), corruption the behavioural columns
  (`robustness_sweep.perturb_derived`). Added as a fifth sweep, `embedding_noise_full`: noisy
  embeddings also rebuild the similarity edges, structural columns, arrival-edge weights and
  coherence columns. Metrics are the whole-sequence score and its final-snapshot node scores;
  online scoring under perturbation is not run (about 8 times the cost), so the sweep tables
  lose their separate window columns.
  *Why amended:* the first version derived the features after the perturbation. Its smoke test
  (four validation sequences) gave every sequence about 0.0005 at noise 0.3, because noise of
  that size swamps the cosines the features are built from. That showed the first version
  differed from the freeze's rule (which does not rebuild the similarity graph under noise).
  Both readings are now reported, so the choice cannot flatter the result. Checks on validation
  sequences: with no perturbation, the tensors and scores are identical to the dataset's. Each
  perturbation changes only its own columns or edges. At noise 0, the full reading rebuilds the
  stored graph exactly.
* *Widened static GAT*: matched to the new detector's 680,515 parameters (hidden 238, 680,943),
  not the frozen 670,915 (`component_ablation_models.matched_static_hidden` now counts the
  detector under the same configuration; the frozen width, 238, is unchanged).
* *Features-only baseline*: the 40 aggregates of `features_only_val.npz`, logistic regression
  fitted on train, read once on test; added to Table 4.4. The oracle comparison gains LR and GB
  on these 40, and LR on 13 + these 40.
* *Not redone*: Table 3.7's controls and the unsupervised baselines (they read the data only,
  which is unchanged); training on the duplicates benchmark (section 6 of the freeze is the record
  of why the benchmark changed; the new detector is cross-scored on it only).

**Running it (28 Oct).** Phase A starts automatically when the step 5 static pilot finishes,
and `src.experiments.coherence_plan` (phases A, B, D1, C, D2, resumable) starts when phase A
does. Progress: `experiments/seeds_coherence/PLAN.md` and `STATUS.md`,
`experiments/rebuild_coherence/STATUS.md`. Test figures go to files only: `train_variant` no
longer prints them, `score_windows --no-metrics` writes scores silently, and every summary's
console output goes to a `.log` beside it.

Code checks before the long runs:
* The modified `seed_tables` and `rebuild` reproduce the freeze from the frozen runs. The
  `seed_tables.json` is byte-identical, and the static-node, record-level, probe and figure
  JSONs are identical.
* `score_online` after the last snapshot equals the whole-window score, graph and node scores
  alike (three validation sequences).
* Every model kind trains one step with the step 5 inputs.
* `oracle_comparison.coherence_aggregates` reproduces `features_only_val.npz` exactly.

**Online scores drift with sequence length (28 Oct, validation, step 5 detector seed 1).** The
end-to-end test of the online path showed that a model trained only on whole 20-snapshot
sequences scores *partial* sequences very differently:

| Read after snapshot | AUROC, poisoned vs clean sequences | Clean, median score | Clean, 95th percentile |
|---|---|---|---|
| 9 | 0.676 | 0.902 | 0.998 |
| 13 | 0.790 | 0.045 | 0.997 |
| 16 | 0.875 | 0.008 | 0.991 |
| 19 (whole sequence) | 0.912 | 0.004 | 0.966 |

Pooled over end points 9 to 19, the online AUROC is 0.837. So one threshold for all end points
(the freeze's latency rule, kept as the primary result) is set by the early, high scores, at
about 0.998, and latency will look poor. **Added, before any test figure:** a secondary latency
block with one threshold per end snapshot, each at the target rate on the benign validation
windows ending there (`seed_tables.per_snapshot_thresholds`, output `latency_per_snapshot`).
This is how a model with this drift would be deployed. The regression keeps its single
threshold. The finding itself goes in the thesis: the gain at sequence level comes with a
detector that is not trained for online use; training on prefixes of random length would be
the fix, and it is not run here.
The online score after snapshot 19 matches the whole-sequence score to within 5e-7 on the GPU,
with no change in rank (identical on the CPU).

**Headline read (30 Oct 19:35, once, as the pre-registration allows after A and B).** Phase A
finished 29 Oct 21:03, phase B 30 Oct 09:26, D1 09:48; the three D1 summaries were identical on
two runs. Test, 500 sequences, five seeds, mean (SD) [95% bootstrap interval]:

| Configuration | Sequence AUROC | AP |
|---|---|---|
| **(C) detector + static GAT, rank mean (chosen by the step 5 rule)** | **0.9221 (0.0089) [0.8975, 0.9424]** | 0.9080 |
| (C), deployable form (validation positions) | 0.9213 (0.0088) [0.8964, 0.9419] | 0.9072 |
| (B) detector alone | 0.9088 (0.0032) [0.8819, 0.9319] | 0.8936 |
| static GAT alone | 0.9156 (0.0120) [0.8916, 0.9367] | 0.8995 |
| Freeze detector, for reference | 0.6614 (0.0144) [0.6166, 0.7054] | |

Paired, test: C minus B +0.0133 [0.0063, 0.0206]; C minus static +0.0065 [0.0001, 0.0134];
detector minus static -0.0068 [-0.0195, 0.0061], not significant. Validation to test, detector:
0.9138 to 0.9088. Online level (Table 4.1's window row): AUROC 0.8248, FPR at 90% TPR 0.565.
Latency at the 5% target: the detector catches 126 to 134 of 200 (benign FPR 3.2% to 4.2%), the
regression 64; per-snapshot thresholds 145 to 156 (4.8% to 6.4%). Size bands 5-8 to 17-20:
0.873 to 0.955. Source `experiments/seeds_coherence/{ensemble_headline,summary_headline,seed_tables}.json`.

**Interruption (noted 2 Nov).** The laptop slept overnight on 31 Oct to 1 Nov (Windows System
log: sleep entered 02:19, return from low power 10:47, after power-source changes), during epoch
2 of `meanpool_s3`, which took 11.5 h of wall time. The run resumed and finished normally
(validation 0.9219, in line with the other mean-pooling seeds), so its result is unaffected, but
its recorded training time (14.9 h, `train_seconds` 53,211) includes the pause and is not a
compute figure; any training-time table uses the other seeds or states the exception.

**Step 6 finished (3 Nov 09:48).** All phases ran; every summary was identical on two runs. All
results, with sources, are in `experiments/audit/RESULTS_FREEZE_2.md` ("freeze 2"). Added
after the last run, descriptive only: `src/experiments/online_drift.py` (the test counterpart
of the drift table above, `S2/online_drift_test.json`), and a second pass of the rebuild steps
from saved scores to check that R2 reproduces (identical except a path field).

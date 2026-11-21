# Notebooks

The method of the thesis as notebooks, in the order a reader meets it. The first two are saved
with their outputs, so they can be read without running anything; the third has to be run.

| Notebook | What it holds | Run time on a Quadro P2000 |
|---|---|---|
| `01_benchmark.ipynb` | The code that builds the benchmark `ap_burst2k_nodup` from AgentPoison's knowledge base. Regenerates all 2,000 sequences and compares every array of every snapshot with the corpus the thesis used. | about 5 min |
| `02_detector.ipynb` | The model inputs (structural features, arrival edges, coherence features), the temporal detector, the static GAT and the ablations, the training loop and the evaluation. Retrains any configuration from scratch. Its last section checks the code against the repository's own modules. | about 20 min as saved; 2.5 to 6 h per model retrained |
| `03_reproduce_freeze2.ipynb` | Every table and figure of Chapter 4 recomputed from the saved scores of the 40 trained models, by the scripts that produced them, and compared with the frozen files. | about 1 h |

## Opening them

From the repository root, with the environment of the main `README.md`:

```
.venv312\Scripts\activate
jupyter lab notebooks
```

The notebooks find the repository root themselves. They need the corpus in
`data/raw/ap_burst2k_nodup` (`01_benchmark.ipynb` writes it if it is missing) and, for the
results, the trained models in `experiments/seeds_coherence/`.

## Retraining

In `02_detector.ipynb`, section 1:

* `TRAIN_MODELS = ["tgn", "static_attn"]` trains the detector and the static GAT of seed `SEED`
  (the pair behind the combined detector), then scores them on the validation and test
  partitions and sets them beside the five reported seeds. Any configuration listed in `MODELS`
  can be named.
* `QUICK_DEMO = True` runs three epochs on 100 sequences (about 3 min), to show training working.

Models trained from the notebook go to `experiments/notebook_runs/` (not tracked by Git). A
finished model is not trained again; delete its folder to repeat it. Training on a GPU is not
bit-reproducible, so a retrained model's scores differ slightly from the reported model of the same
seed; the notebook compares it with the spread of the five reported seeds. On one CPU thread,
where the arithmetic is deterministic, section 7 shows that the notebook's training code and the
repository's give identical losses and weights.

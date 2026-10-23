"""
R2: rebuild the single-checkpoint results on the five-seed, memory-fixed detectors.
==================================================================================

Tables 4.6 and 4.7 to 4.10, Figures 4.1 to 4.3 and the static GAT's record-level row of
Table 4.4 still came from the old single checkpoint (shared memory slots). This rebuilds each
from the five-seed models in --seeds-root, on the chosen main benchmark, and writes one JSON
per table or figure to experiments/rebuild/ with a STATUS.md.

Subcommands:
  static-nodes  Table 4.4, static GAT record-level AUROC and AP, five seeds (from saved scores)
  record-level  record-level AUROC against all records, background records and the matched
                benign clusters, for the detector and the static GAT
  probes        Table 4.6, the four attribution probes, plus the five-seed detector row
  figures       Figures 4.1 to 4.3 from the median seed by sequence AUROC, with source JSON
  sweeps        Tables 4.7 to 4.10, full test partition, three draws, all five seeds (GPU)
  sweep-figures Figures 4.4 to 4.7 from the sweep table, mean +/- SD, with source JSON
  run           all of the above in that order, resumable, keeping the laptop awake

Usage:
    python -m src.experiments.rebuild run --data data/raw/ap_burst2k --seeds-root experiments/seeds
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "experiments" / "rebuild"
T = 10
PY = sys.executable
SWEEP_KEYS = ["graph_auroc", "graph_ap", "graph_f1", "fpr_at_tpr90", "node_auroc", "node_ap",
              "sequence_auroc"]


def status(done, current, notes=""):
    steps = ["static-nodes", "record-level", "probes", "figures", "sweeps"]
    lines = [f"# R2 rebuild: status (updated {dt.datetime.now():%a %d %b %H:%M})", ""]
    for s in steps:
        state = "done" if s in done else ("running" if s == current else "queued")
        lines.append(f"- {s}: **{state}**")
    if notes:
        lines += ["", notes]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def keep_awake(on):
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 if on else 0))


def seed_runs(seeds_root: Path, model: str):
    return sorted(seeds_root.glob(f"{model}_s*/window_scores_test_b1.npz"),
                  key=lambda p: int(p.parent.name.rsplit("_s", 1)[1]))


def seq_auroc(npz):
    from sklearn.metrics import roc_auc_score
    z = np.load(npz, allow_pickle=True)
    ids = sorted(set(z["seq"]))
    y = np.array([int(i.startswith("poisoned")) for i in ids])
    return roc_auc_score(y, [z["score"][z["seq"] == i].max() for i in ids])


def mean_sd(v):
    v = [float(x) for x in v]
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "values": v}


# --------------------------------------------------------------------------- static nodes
def static_nodes(seeds_root: Path):
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = {}
    for p in seed_runs(seeds_root, "static_attn"):
        z = np.load(p, allow_pickle=True)
        rows[p.parent.name] = {"node_auroc": float(roc_auc_score(z["node_label"], z["node_score"])),
                               "node_ap": float(average_precision_score(z["node_label"], z["node_score"]))}
    out = {"source": [str(p) for p in seed_runs(seeds_root, "static_attn")], "per_seed": rows,
           "node_auroc": mean_sd([r["node_auroc"] for r in rows.values()]),
           "node_ap": mean_sd([r["node_ap"] for r in rows.values()])}
    json.dump(out, open(OUT / "table_4_4_static_nodes.json", "w"), indent=2)
    print(f"static GAT node AUROC {out['node_auroc']['mean']:.4f} (SD {out['node_auroc']['sd']:.4f}), "
          f"node AP {out['node_ap']['mean']:.4f}")


# --------------------------------------------------------------------------- record level
def record_level(data: Path, seeds_root: Path):
    """Record-level AUROC three ways, for the detector and the static GAT, five seeds each.

    All records is the figure the thesis reports. It is mostly a single-snapshot task (is this
    record in a dense cluster?), so it is also given against ordinary background records and,
    the comparison that matters, against the records of the matched benign clusters in clean
    sequences, which are as dense as an attack. Cluster positions come from the generator's
    per-sequence RNG (validated against the stored labels: 800/800 identical).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from src.experiments.duplicates_study import replay_slots
    out = {"data": str(data)}
    for model in ("tgn", "static_attn"):
        rows = []
        for p in seed_runs(seeds_root, model):
            z = np.load(p, allow_pickle=True)
            nw, lab, sc = z["node_window"], z["node_label"].astype(bool), z["node_score"]
            wseq = np.asarray(z["seq"])
            in_cluster = np.zeros(len(nw), bool)
            bounds = np.r_[0, np.flatnonzero(np.diff(nw)) + 1, len(nw)]
            slot_cache = {}
            for a, b in zip(bounds[:-1], bounds[1:]):
                sid = wseq[nw[a]]
                if sid not in slot_cache:
                    slot_cache[sid] = replay_slots(sid)[1]
                in_cluster[a:b] = np.isin(np.arange(b - a), slot_cache[sid])
            clean = np.char.startswith(wseq[nw].astype(str), "clean")
            benign_cluster, background = clean & in_cluster, ~in_cluster & ~lab
            m1, m2 = lab | background, lab | benign_cluster
            rows.append({"run": p.parent.name,
                         "all_records": float(roc_auc_score(lab, sc)),
                         "all_records_ap": float(average_precision_score(lab, sc)),
                         "vs_background": float(roc_auc_score(lab[m1], sc[m1])),
                         "vs_matched_cluster": float(roc_auc_score(lab[m2], sc[m2]))})
        out[model] = {"per_seed": rows, **{k: mean_sd([r[k] for r in rows]) for k in
                                           ("all_records", "all_records_ap", "vs_background",
                                            "vs_matched_cluster")}}
    json.dump(out, open(OUT / "table_4_4_record_level.json", "w"), indent=2)
    for model in ("tgn", "static_attn"):
        o = out[model]
        print(f"{model:12s} all {o['all_records']['mean']:.4f}  vs background "
              f"{o['vs_background']['mean']:.4f}  vs matched cluster {o['vs_matched_cluster']['mean']:.4f}")


# --------------------------------------------------------------------------- probes
def probes(data: Path, seeds_root: Path):
    raw = OUT / "probes_raw.json"
    if not raw.exists():
        subprocess.run([PY, "-u", "-m", "src.experiments.separability_probe", "--data", str(data),
                        "--output", str(raw)], check=True, cwd=ROOT)
    from sklearn.metrics import average_precision_score, roc_auc_score
    det = {"window_auroc": [], "window_ap": []}
    for p in seed_runs(seeds_root, "tgn"):
        z = np.load(p, allow_pickle=True)
        det["window_auroc"].append(roc_auc_score(z["y"], z["score"]))
        det["window_ap"].append(average_precision_score(z["y"], z["score"]))
    out = {"data": str(data), "probes": json.load(open(raw)),
           "detector_window_level": {k: mean_sd(v) for k, v in det.items()},
           "detector_source": [str(p) for p in seed_runs(seeds_root, "tgn")]}
    json.dump(out, open(OUT / "table_4_6_probes.json", "w"), indent=2)
    pr = out["probes"]
    print(f"structural {pr['structural']['auroc']:.4f}  content {pr['content']['auroc']:.4f}  "
          f"combined {pr['combined']['auroc']:.4f}  per-record {pr['node_content']['auroc']:.4f}  "
          f"detector {out['detector_window_level']['window_auroc']['mean']:.4f}")


# --------------------------------------------------------------------------- figures
def pick_sequence(te, onset_of, meta, graph_thr):
    """The sequence the figures are drawn on, chosen by a rule fixed before looking at plots.

    Candidates: poisoned test sequences of the median attack size whose onset falls at
    snapshot T+2 to 16, so pre-onset and post-onset windows are both visible. Among those the
    detector catches at the operating point, take the one whose latency equals the median
    latency over all detected test attacks; ties go to the lowest sequence id. If no candidate
    is detected, the first candidate is returned and the caller reports it as a miss.
    """
    seq, start, score = np.asarray(te["seq"]), np.asarray(te["start"]), np.asarray(te["score"])
    pois = sorted(set(s for s in seq if s.startswith("poisoned")))

    def latency(sid):
        w = np.flatnonzero(seq == sid)
        ends = start[w] + T - 1
        hit = np.flatnonzero((ends >= onset_of[sid]) & (score[w] >= graph_thr))
        return int(ends[hit].min() - onset_of[sid]) if hit.size else None

    lat = {s: latency(s) for s in pois}
    detected = [v for v in lat.values() if v is not None]
    med_lat = float(np.median(detected)) if detected else None
    sizes = [meta[s]["num_injected"] for s in pois]
    med_k = int(np.median(sizes))
    cands = [s for s in pois if meta[s]["num_injected"] == med_k and T + 2 <= onset_of[s] <= 16]
    hits = [s for s in cands if lat[s] is not None and lat[s] == med_lat]
    if not hits:
        hits = sorted((s for s in cands if lat[s] is not None),
                      key=lambda s: (abs(lat[s] - med_lat), s)) if med_lat is not None else []
    rule = {"median_attack_size": med_k, "median_latency_detected": med_lat,
            "candidates": len(cands), "candidates_detected": sum(lat[s] is not None for s in cands)}
    return (hits[0] if hits else cands[0]), rule


def figures(data: Path, seeds_root: Path, seq_id: str, fpr: float = 0.05, fig_dir: str = "figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10})

    runs = seed_runs(seeds_root, "tgn")
    aurocs = {p.parent.name: seq_auroc(p) for p in runs}
    ranked = sorted(aurocs, key=aurocs.get)
    median_run = ranked[len(ranked) // 2]
    run_dir = seeds_root / median_run
    te = np.load(run_dir / "window_scores_test_b1.npz", allow_pickle=True)
    va = np.load(run_dir / "window_scores_val_b1.npz", allow_pickle=True)

    # thresholds at the stated benign false-alarm rate, from the whole validation partition
    graph_thr = float(np.quantile(va["score"][va["y"] == 0], 1 - fpr))
    node_thr = float(np.quantile(va["node_score"][va["node_label"] == 0], 1 - fpr))

    meta = {m["sequence_id"]: m for m in json.load(open(data / "metadata.json"))}
    rule = None
    if seq_id == "auto":
        onset_of = {}
        for sid in sorted(set(te["seq"])):
            if sid.startswith("poisoned"):
                with np.load(data / "_seqcache" / "struct" / f"{sid}.npz") as z:
                    onset_of[sid] = int(np.flatnonzero(z["graph_label"])[0])
        seq_id, rule = pick_sequence(te, onset_of, meta, graph_thr)
    seq, start = np.asarray(te["seq"]), np.asarray(te["start"])
    w = np.flatnonzero(seq == seq_id)
    w = w[np.argsort(start[w])]
    graph_scores = te["score"][w].astype(float)
    graph_truth = te["y"][w].astype(int)
    last = w[-1]
    nm = te["node_window"] == last
    node_scores, node_labels = te["node_score"][nm].astype(float), te["node_label"][nm].astype(int)
    with np.load(data / "_seqcache" / "struct" / f"{seq_id}.npz") as z:
        onset = int(np.flatnonzero(z["graph_label"])[0])
        e0, e1 = z["edge_off"][-2], z["edge_off"][-1]
        edges = z["edge_index"][:, e0:e1]
    common = {"run": median_run, "selection_rule": rule or "sequence named on the command line", "seed_sequence_aurocs": aurocs, "sequence": seq_id,
              "injected_records": meta[seq_id]["num_injected"], "onset_snapshot": onset,
              "graph_threshold": graph_thr, "node_threshold": node_thr, "false_alarm_rate": fpr,
              "thresholds_from": "all benign validation windows / records of the same run"}
    figdir = OUT / fig_dir
    figdir.mkdir(parents=True, exist_ok=True)

    # Figure 4.1
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    bn, pz = node_scores[node_labels == 0], node_scores[node_labels == 1]
    bins = np.linspace(0, 1, 36)
    ax.hist(bn, bins=bins, alpha=0.75, color="#3572b0", label=f"Benign (n={len(bn)})")
    ax.hist(pz, bins=bins, alpha=0.75, color="#c0392b", label=f"Poisoned (n={len(pz)})")
    ax.axvline(node_thr, color="gray", ls="--", lw=1.2, label=f"Threshold {node_thr:.3f} ({fpr:.0%} FPR)")
    ax.set_xlabel("Node anomaly score"); ax.set_ylabel("Count")
    ax.set_title(f"Node-level score distribution, final window of {seq_id}")
    ax.legend(fontsize=8.5); plt.tight_layout()
    plt.savefig(figdir / "fig_4_1_node_score_distribution.png", bbox_inches="tight"); plt.close(fig)
    pred = node_scores > node_thr
    json.dump({**common, "node_scores": node_scores.tolist(), "node_labels": node_labels.tolist(),
               "flagged": int(pred.sum()), "true_positives": int((pred & (node_labels == 1)).sum()),
               "false_positives": int((pred & (node_labels == 0)).sum())},
              open(figdir / "fig_4_1_node_score_distribution.json", "w"), indent=1)

    # Figure 4.2
    import networkx as nx
    from matplotlib.patches import Patch
    G = nx.Graph()
    G.add_nodes_from(range(len(node_labels)))
    G.add_edges_from({(int(a), int(b)) for a, b in zip(edges[0], edges[1]) if a != b})
    pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(max(1, G.number_of_nodes())))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, colour_by, title in ((axes[0], node_labels == 1, f"Ground truth: {int(node_labels.sum())} poisoned"),
                                 (axes[1], pred, f"Predicted: {int(pred.sum())} flagged")):
        cols = ["#c0392b" if colour_by[n] else "#3572b0" for n in G.nodes]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cols, node_size=42, linewidths=0)
        ax.set_title(title, fontweight="bold", fontsize=11); ax.axis("off")
    tp = int((pred & (node_labels == 1)).sum()); fp = int((pred & (node_labels == 0)).sum())
    fn = int((~pred & (node_labels == 1)).sum())
    fig.suptitle(f"Node anomalies on the final snapshot of {seq_id}    "
                 f"(TP {tp}, FP {fp}, FN {fn}; threshold {node_thr:.3f})", fontsize=10)
    fig.legend(handles=[Patch(facecolor="#3572b0", label="Benign / not flagged"),
                        Patch(facecolor="#c0392b", label="Poisoned / flagged")],
               loc="lower center", ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(figdir / "fig_4_2_node_graph_overlay.png", bbox_inches="tight"); plt.close(fig)
    json.dump({**common, "edges": edges.tolist(), "node_labels": node_labels.tolist(),
               "flagged": pred.astype(int).tolist(), "tp": tp, "fp": fp, "fn": fn,
               "layout": "networkx spring_layout, seed 42"},
              open(figdir / "fig_4_2_node_graph_overlay.json", "w"))

    # Figure 4.3
    fig, ax = plt.subplots(figsize=(11, 4.2))
    steps = start[w] + T - 1
    ax.fill_between(steps, 0, graph_truth, alpha=0.18, color="#c0392b", step="mid",
                    label="Ground truth (injection present)")
    ax.plot(steps, graph_scores, "-o", color="#1f4e79", markersize=4.5, label="Detector score")
    ax.axhline(graph_thr, color="gray", ls="--", lw=1, label=f"Window operating point ({fpr:.0%} FPR)")
    ax.axvline(onset, color="#c0392b", ls=":", lw=1.4, label=f"Attack onset (snapshot {onset})")
    fired = np.flatnonzero((graph_scores >= graph_thr) & (steps >= onset))
    if fired.size:
        ax.axvline(steps[fired[0]], color="#2e7d32", ls="-.", lw=1.4, label="First alert")
    ax.set_xlabel("Snapshot index (final snapshot of each window)"); ax.set_ylabel("Poisoning score")
    ax.set_title(f"Detection score across a flood-placement sequence ({seq_id})")
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.25); ax.legend(loc="upper left", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(figdir / "fig_4_3_single_sequence_detection_score.png", bbox_inches="tight"); plt.close(fig)
    json.dump({**common, "window_end_snapshot": steps.tolist(), "graph_scores": graph_scores.tolist(),
               "graph_truth": graph_truth.tolist(),
               "first_alert_snapshot": int(steps[fired[0]]) if fired.size else None},
              open(figdir / "fig_4_3_single_sequence_detection_score.json", "w"), indent=1)
    summary = {**common, "files": sorted(p.name for p in figdir.iterdir())}
    json.dump(summary, open(OUT / f"{fig_dir}_4_1_to_4_3.json", "w"), indent=2)
    print(f"figures from {median_run} (median of {[round(v, 4) for v in sorted(aurocs.values())]}), "
          f"sequence {seq_id}: {tp} of {int(node_labels.sum())} injected flagged, {fp} benign flagged; "
          f"first alert at snapshot {int(steps[fired[0]]) if fired.size else None} (onset {onset})")


# --------------------------------------------------------------------------- sweep figures
def sweep_figures():
    """Figures 4.4 to 4.7 from tables_4_7_to_4_10_sweeps.json: mean +/- SD over seeds x draws."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10})
    src = json.load(open(OUT / "tables_4_7_to_4_10_sweeps.json"))
    table = src["table"]
    lines = (("graph_auroc", "Graph AUROC (window)", "#1f4e79", "o", "-"),
             ("graph_ap", "Graph AP (window)", "#c0392b", "s", "-"),
             ("node_auroc", "Node AUROC", "#2e7d32", "^", "-"),
             ("sequence_auroc", "Sequence AUROC", "#6a3d9a", "D", "--"))
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    note = (f"mean +/- SD over {len(src['runs'])} seeds x {src['draws_per_condition']} draws, "
            f"{src['n_windows']:,} test windows")

    def draw(ax, name, xlabel, title):
        rows = table[name]
        xs = sorted(rows, key=float)
        for key, label, colour, mark, ls in lines:
            m = np.array([rows[x][key]["mean"] for x in xs])
            sd = np.array([rows[x][key]["sd"] or 0.0 for x in xs])
            ax.errorbar([float(x) for x in xs], m, yerr=sd, marker=mark, color=colour, ls=ls,
                        label=label, markersize=5, capsize=3, lw=1.4)
        ax.set_xlabel(xlabel); ax.set_ylabel("Score"); ax.set_title(title, fontsize=10.5)
        ax.set_ylim(0.35, 0.85); ax.grid(alpha=0.25); ax.legend(fontsize=8)

    specs = {"4_4": [("embedding_noise", "Gaussian noise standard deviation",
                      "Detection performance under embedding noise")],
             "4_5": [("edge_threshold", "Cosine similarity edge threshold",
                      "Detection performance under variation of the edge threshold")],
             "4_6": [("metadata_corruption", "Metadata corruption probability", "Metadata corruption"),
                     ("edge_loss", "Edge drop probability", "Random edge loss")],
             "4_7": [("embedding_noise", "Noise sigma", "Embedding noise"),
                     ("edge_threshold", "Edge threshold", "Edge construction threshold"),
                     ("metadata_corruption", "Corruption probability", "Metadata corruption"),
                     ("edge_loss", "Drop probability", "Random edge loss")]}
    names = {"4_4": "fig_4_4_embedding_noise", "4_5": "fig_4_5_edge_threshold",
             "4_6": "fig_4_6_metadata_and_edge_loss", "4_7": "fig_4_7_degradation_dashboard"}
    for fig_id, panels in specs.items():
        if len(panels) == 1:
            fig, axes = plt.subplots(figsize=(7.5, 4.4)); axes = [axes]
        elif len(panels) == 2:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12.5, 8)); axes = axes.ravel()
        for ax, (name, xlabel, title) in zip(axes, panels):
            draw(ax, name, xlabel, title)
        if fig_id == "4_5":
            axes[0].axvline(0.6, color="gray", ls=":", lw=1.2)
            axes[0].text(0.6, 0.37, " trained value", fontsize=8, color="gray")
        if fig_id == "4_6":
            fig.suptitle(f"Degraded operational logging ({note})", fontsize=10.5)
        if fig_id == "4_7":
            fig.suptitle(f"Detector performance under four input degradations "
                         f"(ap_burst2k_nodup; {note})", fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95] if fig_id in ("4_6", "4_7") else None)
        plt.savefig(figdir / f"{names[fig_id]}.png", bbox_inches="tight"); plt.close(fig)
        json.dump({"source": "experiments/rebuild/tables_4_7_to_4_10_sweeps.json", "note": note,
                   "panels": {n: table[n] for n, _, _ in panels}},
                  open(figdir / f"{names[fig_id]}.json", "w"), indent=1)
    print(f"Figures 4.4 to 4.7 written to {figdir} ({note})")


# --------------------------------------------------------------------------- sweeps
def sweeps(data: Path, seeds_root: Path, draws: int):
    sdir = OUT / "sweeps"
    sdir.mkdir(parents=True, exist_ok=True)
    runs = seed_runs(seeds_root, "tgn")
    for i, p in enumerate(runs):
        status(["static-nodes", "record-level", "probes", "figures"], "sweeps",
               f"Sweeps: seed run {i + 1} of {len(runs)} ({p.parent.name}); about 3 h per run.")
        subprocess.run([PY, "-u", "-m", "src.experiments.robustness_sweep", "--data", str(data),
                        "--run-dir", str(p.parent), "--draws", str(draws),
                        "--output", str(sdir / f"{p.parent.name}.json")], check=True, cwd=ROOT)
    per = [json.load(open(sdir / f"{p.parent.name}.json")) for p in runs]
    table = {}
    for name in per[0]["sweeps"]:
        table[name] = {}
        for v in per[0]["sweeps"][name]:
            vals = {k: [d[k] for r in per for d in r["sweeps"][name][v]["draws"]] for k in SWEEP_KEYS}
            table[name][v] = {k: mean_sd(x) for k, x in vals.items()}
            table[name][v]["n_values"] = len(vals["graph_auroc"])
    out = {"data": str(data), "runs": [p.parent.name for p in runs], "draws_per_condition": draws,
           "n_windows": per[0]["n_windows"], "n_sequences": per[0]["n_sequences"],
           "memory_slots": "per_window", "mean_and_sd_over": "seeds x draws", "table": table}
    json.dump(out, open(OUT / "tables_4_7_to_4_10_sweeps.json", "w"), indent=2)
    for name, rows in table.items():
        print(name, {v: round(r["graph_auroc"]["mean"], 4) for v, r in rows.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["static-nodes", "record-level", "probes", "figures",
                                        "sweeps", "sweep-figures", "run"])
    ap.add_argument("--data", default="data/raw/ap_burst2k")
    ap.add_argument("--seeds-root", default="experiments/seeds")
    ap.add_argument("--sequence", default="auto",
                    help="Test sequence for Figures 4.1 to 4.3: 'auto' applies the rule in "
                         "pick_sequence; or name one (the thesis used poisoned_placement_0463).")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", default=None, help="Output folder (default experiments/rebuild)")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = Path(args.out)
    sys.path.insert(0, str(ROOT))
    data, seeds_root = ROOT / args.data, ROOT / args.seeds_root
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "static-nodes":
        static_nodes(seeds_root)
    elif args.command == "record-level":
        record_level(data, seeds_root)
    elif args.command == "probes":
        probes(data, seeds_root)
    elif args.command == "figures":
        figures(data, seeds_root, args.sequence, fig_dir=args.fig_dir)
    elif args.command == "sweeps":
        sweeps(data, seeds_root, args.draws)
    elif args.command == "sweep-figures":
        sweep_figures()
    else:
        keep_awake(True)
        try:
            done = []
            for name, fn in (("static-nodes", lambda: static_nodes(seeds_root)),
                             ("record-level", lambda: record_level(data, seeds_root)),
                             ("probes", lambda: probes(data, seeds_root)),
                             ("figures", lambda: figures(data, seeds_root, args.sequence)),
                             ("sweeps", lambda: sweeps(data, seeds_root, args.draws))):
                status(done, name)
                fn()
                done.append(name)
            status(done, None, "All R2 steps finished.")
        finally:
            keep_awake(False)


if __name__ == "__main__":
    main()

"""
Visualize Memory Graph Datasets
================================
Generates visual comparisons of clean vs poisoned memory graphs.

Usage:
    python -m src.utils.visualize --data data/raw/llm_generated
    python -m src.utils.visualize --data data/raw/llm_generated --sequence poisoned_minja_query_only_0000
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from matplotlib.patches import Patch


def load_snapshot(path):
    """Load a single .npz snapshot."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def snapshot_to_networkx(snap):
    """Convert snapshot arrays to a NetworkX graph."""
    G = nx.Graph()
    n = snap["node_features"].shape[0]

    for i in range(n):
        G.add_node(i, poisoned=int(snap["node_labels"][i]))

    edge_index = snap["edge_index"]
    edge_weights = snap["edge_weights"]
    if edge_index.shape[1] > 0:
        for idx in range(0, edge_index.shape[1], 2):  # Skip duplicates (undirected)
            src, dst = edge_index[0, idx], edge_index[1, idx]
            w = edge_weights[idx] if idx < len(edge_weights) else 0.5
            G.add_edge(int(src), int(dst), weight=float(w))

    return G


def plot_single_graph(ax, G, title):
    """Plot a single memory graph on the given axis."""
    if len(G.nodes) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center", transform=ax.transAxes)
        return

    # Color nodes: blue=benign, red=poisoned
    colors = ["#e74c3c" if G.nodes[n].get("poisoned", 0) == 1 else "#3498db" for n in G.nodes]

    # Layout
    if len(G.nodes) < 100:
        pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(len(G.nodes)))
    else:
        pos = nx.spring_layout(G, seed=42, iterations=30)

    # Draw edges (thin, gray)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5, edge_color="#bdc3c7")

    # Draw nodes
    node_sizes = [30 if len(G.nodes) > 50 else 80]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=node_sizes[0], alpha=0.8)

    ax.set_title(title, fontsize=10)
    ax.axis("off")


def visualize_sequence(data_dir, seq_id, output_path=None):
    """Visualize snapshots from a single sequence over time."""
    seq_dir = Path(data_dir) / seq_id
    snapshots = sorted(seq_dir.glob("snapshot_*.npz"))

    if not snapshots:
        print(f"No snapshots found in {seq_dir}")
        return

    # Pick evenly spaced snapshots (max 8)
    n = len(snapshots)
    indices = np.linspace(0, n - 1, min(8, n), dtype=int)
    selected = [snapshots[i] for i in indices]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (snap_path, ax) in enumerate(zip(selected, axes)):
        snap = load_snapshot(snap_path)
        G = snapshot_to_networkx(snap)
        n_poisoned = int(snap["node_labels"].sum())
        n_total = len(snap["node_labels"])
        label = "POISONED" if snap["graph_label"][0] == 1 else "CLEAN"
        title = f"t={indices[i]} | {label}\n{n_total} nodes, {n_poisoned} poisoned"
        plot_single_graph(ax, G, title)

    # Hide unused axes
    for i in range(len(selected), len(axes)):
        axes[i].axis("off")

    # Legend
    legend_elements = [
        Patch(facecolor="#3498db", label="Benign record"),
        Patch(facecolor="#e74c3c", label="Poisoned record"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=11)

    fig.suptitle(f"Memory Graph Evolution: {seq_id}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()


def visualize_comparison(data_dir, output_path=None):
    """Side-by-side comparison of a clean vs poisoned sequence (final snapshot)."""
    with open(Path(data_dir) / "metadata.json") as f:
        metadata = json.load(f)

    clean_seqs = [m for m in metadata if m["label"] == "clean"]
    poisoned_seqs = [m for m in metadata if m["label"] == "poisoned"]

    # Pick one of each
    clean_id = clean_seqs[0]["sequence_id"]
    poisoned_id = poisoned_seqs[0]["sequence_id"]

    # Load last snapshot from each
    clean_snaps = sorted((Path(data_dir) / clean_id).glob("snapshot_*.npz"))
    poisoned_snaps = sorted((Path(data_dir) / poisoned_id).glob("snapshot_*.npz"))

    clean_snap = load_snapshot(clean_snaps[-1])
    poisoned_snap = load_snapshot(poisoned_snaps[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    G_clean = snapshot_to_networkx(clean_snap)
    G_poisoned = snapshot_to_networkx(poisoned_snap)

    n_clean = len(clean_snap["node_labels"])
    n_poisoned_nodes = int(poisoned_snap["node_labels"].sum())
    n_total_poisoned = len(poisoned_snap["node_labels"])

    plot_single_graph(ax1, G_clean, f"CLEAN: {clean_id}\n{n_clean} nodes, 0 poisoned")
    plot_single_graph(ax2, G_poisoned, f"POISONED: {poisoned_id}\n{n_total_poisoned} nodes, {n_poisoned_nodes} poisoned")

    legend_elements = [
        Patch(facecolor="#3498db", label="Benign record"),
        Patch(facecolor="#e74c3c", label="Poisoned record"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=11)

    fig.suptitle("Clean vs Poisoned Memory Graph (Final Snapshot)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()


def visualize_stats(data_dir, output_path=None):
    """Plot dataset statistics: node counts, edge density, poisoned ratio over time."""
    with open(Path(data_dir) / "metadata.json") as f:
        metadata = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Collect stats
    clean_nodes, poisoned_nodes = [], []
    clean_edges, poisoned_edges = [], []

    for m in metadata:
        seq_dir = Path(data_dir) / m["sequence_id"]
        snaps = sorted(seq_dir.glob("snapshot_*.npz"))
        if not snaps:
            continue
        last = load_snapshot(snaps[-1])
        n_nodes = last["node_features"].shape[0]
        n_edges = last["edge_index"].shape[1] // 2 if last["edge_index"].shape[1] > 0 else 0

        if m["label"] == "clean":
            clean_nodes.append(n_nodes)
            clean_edges.append(n_edges)
        else:
            poisoned_nodes.append(n_nodes)
            poisoned_edges.append(n_edges)

    # Plot 1: Node count distribution
    axes[0, 0].hist(clean_nodes, bins=20, alpha=0.7, label="Clean", color="#3498db")
    axes[0, 0].hist(poisoned_nodes, bins=20, alpha=0.7, label="Poisoned", color="#e74c3c")
    axes[0, 0].set_xlabel("Number of nodes (final snapshot)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Node Count Distribution")
    axes[0, 0].legend()

    # Plot 2: Edge count distribution
    axes[0, 1].hist(clean_edges, bins=20, alpha=0.7, label="Clean", color="#3498db")
    axes[0, 1].hist(poisoned_edges, bins=20, alpha=0.7, label="Poisoned", color="#e74c3c")
    axes[0, 1].set_xlabel("Number of edges (final snapshot)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Edge Count Distribution")
    axes[0, 1].legend()

    # Plot 3: Graph evolution over time (one poisoned sequence)
    poisoned_meta = [m for m in metadata if m["label"] == "poisoned"][0]
    seq_dir = Path(data_dir) / poisoned_meta["sequence_id"]
    snaps = sorted(seq_dir.glob("snapshot_*.npz"))

    timestamps, node_counts, poisoned_counts, edge_counts = [], [], [], []
    for snap_path in snaps:
        snap = load_snapshot(snap_path)
        timestamps.append(snap["timestamp"][0])
        node_counts.append(snap["node_features"].shape[0])
        poisoned_counts.append(int(snap["node_labels"].sum()))
        edge_counts.append(snap["edge_index"].shape[1] // 2)

    ax3 = axes[1, 0]
    ax3.plot(timestamps, node_counts, "b-", label="Total nodes")
    ax3.plot(timestamps, poisoned_counts, "r-", label="Poisoned nodes")
    ax3.set_xlabel("Timestamp")
    ax3.set_ylabel("Count")
    ax3.set_title(f"Evolution: {poisoned_meta['sequence_id']}")
    ax3.legend()

    # Plot 4: Attack type breakdown
    attack_types = {}
    for m in metadata:
        if m["label"] == "poisoned":
            at = m.get("attack_type", "unknown")
            attack_types[at] = attack_types.get(at, 0) + 1

    axes[1, 1].bar(attack_types.keys(), attack_types.values(), color=["#e74c3c", "#f39c12", "#9b59b6"])
    axes[1, 1].set_xlabel("Attack Type")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Attack Type Distribution")
    axes[1, 1].tick_params(axis="x", rotation=15)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize memory graph dataset")
    parser.add_argument("--data", type=str, default="data/raw/llm_generated")
    parser.add_argument("--sequence", type=str, default=None, help="Specific sequence ID to visualize")
    parser.add_argument("--mode", type=str, choices=["compare", "sequence", "stats", "all"], default="all")
    parser.add_argument("--save", type=str, default=None, help="Directory to save plots (shows interactively if not set)")
    args = parser.parse_args()

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("compare", "all"):
        out = str(save_dir / "comparison.png") if save_dir else None
        visualize_comparison(args.data, output_path=out)

    if args.mode in ("stats", "all"):
        out = str(save_dir / "stats.png") if save_dir else None
        visualize_stats(args.data, output_path=out)

    if args.mode in ("sequence", "all"):
        if args.sequence:
            seq_id = args.sequence
        else:
            # Pick first poisoned sequence
            with open(Path(args.data) / "metadata.json") as f:
                metadata = json.load(f)
            seq_id = next(m["sequence_id"] for m in metadata if m["label"] == "poisoned")

        out = str(save_dir / f"{seq_id}.png") if save_dir else None
        visualize_sequence(args.data, seq_id, output_path=out)


if __name__ == "__main__":
    main()

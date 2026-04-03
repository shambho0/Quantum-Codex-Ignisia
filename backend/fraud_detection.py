"""
backend/fraud_detection.py
---------------------------
Graph-based fraud detection module using NetworkX.
Detects circular transaction topologies among linked MSMEs
where funds are rotated to artificially inflate scores.

Enhances the notebook's simple fraud_flag with graph analysis.
"""

import hashlib
import networkx as nx
import numpy as np


# ──────────────────────────────────────────────
# Simulated MSME transaction graph builder
# ──────────────────────────────────────────────

def _build_transaction_graph(gstin: str, n_related: int = 10) -> nx.DiGraph:
    """
    Build a directed UPI transaction graph centered on the given GSTIN.
    Related parties are deterministically derived from the GSTIN hash.
    
    In a real system, this would query a live transaction database.
    """
    G = nx.DiGraph()
    seed = int(hashlib.md5(gstin.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    # Generate related GSTIN nodes (simulated business network)
    related = []
    for i in range(n_related):
        node_seed = seed + i
        node_id = f"GSTIN_{node_seed % 10000:05d}"
        related.append(node_id)

    G.add_node(gstin)
    for node in related:
        G.add_node(node)

    # Add transaction edges (directed: sender → receiver)
    # Realistic probabilities — most businesses have clean transaction patterns
    for i, node in enumerate(related):
        flow_amount = int(rng.integers(5000, 100000))

        # Primary GSTIN sends to some party (normal outgoing payments)
        if rng.random() > 0.3:
            G.add_edge(gstin, node, amount=flow_amount)

        # Some parties send back (normal receivables — NOT fraud by itself)
        if rng.random() > 0.65:
            back_amount = int(flow_amount * rng.uniform(0.8, 1.2))
            G.add_edge(node, gstin, amount=back_amount)

        # Cross-transactions between related parties (enables 3+ node cycles)
        # Low probability — circular routing is rare in legitimate business
        if i + 1 < len(related) and rng.random() > 0.82:
            cross_amount = int(rng.integers(5000, 50000))
            G.add_edge(node, related[(i + 1) % len(related)], amount=cross_amount)

    return G


def detect_circular_transactions(gstin: str) -> dict:
    """
    Analyze the transaction graph for circular topologies.
    
    Only cycles of length >= 3 are flagged as suspicious (A→B→C→A).
    Two-node back-and-forth (A↔B) is normal business behaviour
    (paying a supplier who also pays you) and is NOT considered fraud.
    
    Returns:
        is_circular: bool — True if circular patterns detected
        cycle_count: int  — Number of suspicious cycles found (length 3+)
        risk_score: float — 0.0 (clean) to 1.0 (high fraud risk)
        affected_nodes: list[str] — GSTINs involved in cycles
        graph_summary: dict — stats about the transaction graph
    """
    G = _build_transaction_graph(gstin)

    # Find all simple cycles in the directed graph
    try:
        cycles = list(nx.simple_cycles(G))
    except nx.NetworkXError:
        cycles = []

    # Filter: only count cycles that
    #  1. include the primary GSTIN, AND
    #  2. have length >= 3 (skip trivial 2-node back-and-forth)
    primary_cycles = [
        c for c in cycles
        if gstin in c and len(c) >= 3
    ]

    # Collect all nodes involved in suspicious cycles
    affected_nodes = set()
    for cycle in primary_cycles:
        for node in cycle:
            if node != gstin:
                affected_nodes.add(node)

    cycle_count = len(primary_cycles)

    # Risk scoring heuristic:
    # 0 cycles = clean, 1 = low-moderate, 2 = moderate, 3+ = high risk
    if cycle_count == 0:
        risk_score = 0.0
    elif cycle_count == 1:
        risk_score = 0.25
    elif cycle_count == 2:
        risk_score = 0.45
    else:
        risk_score = min(0.9, 0.55 + (cycle_count * 0.1))

    return {
        "is_circular":    cycle_count > 0,
        "cycle_count":    cycle_count,
        "risk_score":     round(risk_score, 3),
        "affected_nodes": sorted(affected_nodes),
        "graph_summary": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "in_degree":   G.in_degree(gstin),
            "out_degree":  G.out_degree(gstin),
        },
    }

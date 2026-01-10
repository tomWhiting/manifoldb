#!/usr/bin/env python3
"""
Visualize a ManifoldDB session graph.

This script reads a ManifoldDB backup JSONL file containing a Claude Code session
and generates an interactive HTML visualization of the conversation graph.

Usage:
    python visualize-session.py <backup.jsonl> [output.html]

Requirements:
    pip install pyvis networkx
"""

import json
import sys
from pathlib import Path

try:
    from pyvis.network import Network
    import networkx as nx
except ImportError:
    print("Missing dependencies. Install with:", file=sys.stderr)
    print("  pip install pyvis networkx", file=sys.stderr)
    sys.exit(1)


def load_graph_from_backup(backup_path: Path) -> nx.DiGraph:
    """Load graph from ManifoldDB backup JSONL file."""
    G = nx.DiGraph()

    entities = {}
    edges = []

    with open(backup_path) as f:
        for line in f:
            record = json.loads(line)
            record_type = record.get("type")

            if record_type == "entity":
                data = record["data"]
                entity_id = data["id"]
                labels = data.get("labels", [])
                props = data.get("properties", {})
                entities[entity_id] = {"labels": labels, "properties": props}

            elif record_type == "edge":
                data = record["data"]
                edges.append({
                    "id": data["id"],
                    "source": data["source"],
                    "target": data["target"],
                    "type": data["edge_type"],
                    "properties": data.get("properties", {}),
                })

    # Add nodes
    for entity_id, entity in entities.items():
        labels = entity["labels"]
        props = entity["properties"]

        # Determine node appearance based on labels
        if "User" in labels:
            color = "#4CAF50"  # Green
            msg_type = "User"
            size = 25
            shape = "dot"
        elif "Assistant" in labels:
            color = "#2196F3"  # Blue
            msg_type = "Assistant"
            size = 25
            shape = "dot"
        elif "QueueOperation" in labels:
            color = "#9E9E9E"  # Gray
            msg_type = "Queue"
            size = 15
            shape = "dot"
        elif "ToolUse" in labels:
            tool_name = props.get("toolName", "Tool")
            tool_colors = {
                "Bash": "#FF5722",      # Deep Orange
                "Read": "#9C27B0",      # Purple
                "Write": "#E91E63",     # Pink
                "Grep": "#00BCD4",      # Cyan
                "Glob": "#009688",      # Teal
                "TodoWrite": "#FFC107", # Amber
            }
            color = tool_colors.get(tool_name, "#607D8B")
            msg_type = tool_name
            size = 18
            shape = "box"
        elif "ToolResult" in labels:
            color = "#795548"  # Brown
            msg_type = "Result"
            size = 12
            shape = "diamond"
        else:
            color = "#666666"
            msg_type = labels[0] if labels else "Unknown"
            size = 15
            shape = "dot"

        # Build label
        uuid = props.get("uuid", "")
        if uuid:
            label = f"{msg_type}\n{uuid[:8]}"
        else:
            label = msg_type

        # Build tooltip
        title_parts = [f"ID: {entity_id}", f"Labels: {', '.join(labels)}"]
        if props.get("timestamp"):
            title_parts.append(f"Time: {props['timestamp']}")
        if props.get("command"):
            cmd = props["command"]
            if len(cmd) > 80:
                cmd = cmd[:80] + "..."
            title_parts.append(f"Command: {cmd}")
        if props.get("textContent"):
            text = props["textContent"]
            if len(text) > 200:
                text = text[:200] + "..."
            title_parts.append(f"Content: {text}")
        if props.get("outputTokens"):
            title_parts.append(f"Output tokens: {props['outputTokens']}")

        G.add_node(
            entity_id,
            label=label,
            color=color,
            title="\n".join(title_parts),
            size=size,
            shape=shape,
        )

    # Add edges
    edge_colors = {
        "FOLLOWS": "#888888",
        "USES_TOOL": "#FF9800",
        "HAS_RESULT": "#795548",
    }

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        edge_type = edge["type"]

        if source in G.nodes and target in G.nodes:
            G.add_edge(
                source,
                target,
                color=edge_colors.get(edge_type, "#666666"),
                title=edge_type,
                width=2 if edge_type == "FOLLOWS" else 1,
            )

    return G


def visualize(G: nx.DiGraph, output_path: Path):
    """Create an interactive HTML visualization."""
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
    )

    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.01,
                "springLength": 150,
                "springConstant": 0.05,
                "damping": 0.4
            },
            "stabilization": {
                "enabled": true,
                "iterations": 300
            }
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true
        },
        "nodes": {
            "font": {
                "size": 12
            }
        }
    }
    """)

    net.from_nx(G)
    net.save_graph(str(output_path))
    print(f"Visualization saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <backup.jsonl> [output.html]", file=sys.stderr)
        print("\nVisualizes a ManifoldDB session graph from a backup file.", file=sys.stderr)
        sys.exit(1)

    backup_path = Path(sys.argv[1])
    if not backup_path.exists():
        print(f"Backup file not found: {backup_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else backup_path.with_suffix(".html")

    print(f"Loading graph from: {backup_path}")
    G = load_graph_from_backup(backup_path)
    print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    visualize(G, output_path)
    print(f"\nOpen in browser: file://{output_path.absolute()}")


if __name__ == "__main__":
    main()

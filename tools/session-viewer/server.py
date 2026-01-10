#!/usr/bin/env python3
"""
Simple API server for querying ManifoldDB session graphs.

Runs on http://localhost:8765 and provides endpoints for:
- GET /api/graph - Returns all nodes and edges
- GET /api/nodes - Returns all nodes
- GET /api/edges - Returns all edges
- GET /api/stats - Returns graph statistics

Usage:
    python server.py <database.manifold>

    # Or with uvicorn directly:
    DATABASE_PATH=/path/to/db.manifold uvicorn server:app --port 8765
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="ManifoldDB Session Viewer API")

# Enable CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path from environment or command line
DATABASE_PATH = os.environ.get("DATABASE_PATH", "")
MANIFOLD_BIN = os.environ.get("MANIFOLD_BIN", "")


def find_manifold_binary() -> Path | None:
    """Find the manifold CLI binary."""
    if MANIFOLD_BIN:
        p = Path(MANIFOLD_BIN)
        if p.exists():
            return p

    # Look in common locations relative to this script
    script_dir = Path(__file__).parent
    search_paths = [
        script_dir.parent.parent / "target" / "release" / "manifold",
        script_dir.parent.parent / "target" / "debug" / "manifold",
        Path("./target/release/manifold"),
        Path("manifold"),
    ]

    for p in search_paths:
        if p.exists():
            return p
    return None


def query_db(cypher: str) -> list[dict]:
    """Execute a Cypher query against the database."""
    if not DATABASE_PATH:
        raise HTTPException(status_code=500, detail="DATABASE_PATH not configured")

    manifold = find_manifold_binary()
    if not manifold:
        raise HTTPException(status_code=500, detail="manifold binary not found")

    result = subprocess.run(
        [str(manifold), "-d", DATABASE_PATH, "query", cypher, "-f", "json"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Query failed: {result.stderr}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON response: {e}")


class Node(BaseModel):
    id: int
    labels: list[str]
    properties: dict


class Edge(BaseModel):
    id: int
    source: int
    target: int
    type: str
    properties: dict


class GraphData(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


class Stats(BaseModel):
    node_count: int
    edge_count: int
    labels: dict[str, int]
    edge_types: dict[str, int]


def load_from_backup() -> GraphData:
    """Load graph data from the backup JSONL file (fallback method)."""
    # Try to find the backup file based on database path
    if not DATABASE_PATH:
        raise HTTPException(status_code=500, detail="DATABASE_PATH not configured")

    db_path = Path(DATABASE_PATH)
    backup_candidates = [
        db_path.parent / f"{db_path.stem}-graph.jsonl",
        db_path.with_suffix(".jsonl"),
    ]

    backup_path = None
    for candidate in backup_candidates:
        if candidate.exists():
            backup_path = candidate
            break

    if not backup_path:
        raise HTTPException(
            status_code=404,
            detail="No backup file found. Run convert-transcript.py first.",
        )

    nodes = []
    edges = []

    with open(backup_path) as f:
        for line in f:
            record = json.loads(line)
            record_type = record.get("type")

            if record_type == "entity":
                data = record["data"]
                nodes.append(Node(
                    id=data["id"],
                    labels=data.get("labels", []),
                    properties=data.get("properties", {}),
                ))

            elif record_type == "edge":
                data = record["data"]
                edges.append(Edge(
                    id=data["id"],
                    source=data["source"],
                    target=data["target"],
                    type=data["edge_type"],
                    properties=data.get("properties", {}),
                ))

    return GraphData(nodes=nodes, edges=edges)


@app.get("/api/graph", response_model=GraphData)
async def get_graph():
    """Get full graph data (nodes and edges)."""
    # For now, use backup file loading since Cypher queries
    # have some limitations with node returns
    return load_from_backup()


@app.get("/api/stats", response_model=Stats)
async def get_stats():
    """Get graph statistics."""
    graph = load_from_backup()

    label_counts: dict[str, int] = {}
    for node in graph.nodes:
        for label in node.labels:
            label_counts[label] = label_counts.get(label, 0) + 1

    edge_type_counts: dict[str, int] = {}
    for edge in graph.edges:
        edge_type_counts[edge.type] = edge_type_counts.get(edge.type, 0) + 1

    return Stats(
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        labels=label_counts,
        edge_types=edge_type_counts,
    )


@app.get("/api/pagerank")
async def get_pagerank():
    """Run PageRank algorithm on the graph."""
    results = query_db("CALL algo.pageRank() YIELD nodeId, score")
    return results


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "database": DATABASE_PATH,
        "manifold_binary": str(find_manifold_binary()),
    }


# Serve static files in production
dist_path = Path(__file__).parent / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <database.manifold>", file=sys.stderr)
        print("\nStarts a web server for visualizing ManifoldDB session graphs.", file=sys.stderr)
        sys.exit(1)

    DATABASE_PATH = sys.argv[1]
    os.environ["DATABASE_PATH"] = DATABASE_PATH

    print(f"Starting server with database: {DATABASE_PATH}")
    print(f"API: http://localhost:8765/api/graph")
    print(f"UI:  http://localhost:5173 (run 'npm run dev' in another terminal)")

    uvicorn.run(app, host="0.0.0.0", port=8765)

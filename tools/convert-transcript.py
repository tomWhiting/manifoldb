#!/usr/bin/env python3
"""
Convert Claude Code session transcripts to ManifoldDB graph format.

This script transforms JSONL session transcripts into a graph structure:

Entities:
- Message (labels: User, Assistant, QueueOperation)
- ToolUse (extracted from message content)
- ToolResult (extracted from message content)

Edges:
- FOLLOWS: Message -> Message (via parentUuid)
- USES_TOOL: Message -> ToolUse
- HAS_RESULT: ToolUse -> ToolResult

Usage:
    python convert-transcript.py input.jsonl output.jsonl
    python convert-transcript.py input.jsonl  # outputs to stdout

The output is in ManifoldDB backup JSONL format.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any


def generate_id(counter: list[int]) -> int:
    """Generate a monotonically increasing ID."""
    counter[0] += 1
    return counter[0]


def create_metadata_record(stats: dict[str, int]) -> dict:
    """Create backup metadata record."""
    return {
        "type": "metadata",
        "data": {
            "version": 1,
            "format": "json_lines",
            "created_at": int(time.time()),
            "sequence_number": 0,
            "is_incremental": False,
            "previous_sequence": None,
            "statistics": {
                "entity_count": stats.get("entities", 0),
                "edge_count": stats.get("edges", 0),
                "metadata_count": 0,
                "total_records": stats.get("entities", 0) + stats.get("edges", 0),
                "uncompressed_size": 0,
            },
        },
    }


def create_entity_record(
    entity_id: int, labels: list[str], properties: dict[str, Any]
) -> dict:
    """Create an entity record in backup format."""
    # Convert properties to ManifoldDB-compatible format
    clean_props = {}
    for key, value in properties.items():
        if value is None:
            continue  # Skip null values
        elif isinstance(value, (dict, list)):
            # Serialize complex objects as JSON strings
            clean_props[key] = json.dumps(value)
        elif isinstance(value, bool):
            clean_props[key] = value
        elif isinstance(value, int):
            clean_props[key] = value
        elif isinstance(value, float):
            clean_props[key] = value
        else:
            clean_props[key] = str(value)

    return {
        "type": "entity",
        "data": {
            "id": entity_id,
            "labels": labels,
            "properties": clean_props,
        },
    }


def create_edge_record(
    edge_id: int,
    source_id: int,
    target_id: int,
    edge_type: str,
    properties: dict[str, Any] | None = None,
) -> dict:
    """Create an edge record in backup format."""
    clean_props = {}
    if properties:
        for key, value in properties.items():
            if value is None:
                continue
            elif isinstance(value, (dict, list)):
                clean_props[key] = json.dumps(value)
            elif isinstance(value, bool):
                clean_props[key] = value
            elif isinstance(value, int):
                clean_props[key] = value
            elif isinstance(value, float):
                clean_props[key] = value
            else:
                clean_props[key] = str(value)

    return {
        "type": "edge",
        "data": {
            "id": edge_id,
            "source": source_id,
            "target": target_id,
            "edge_type": edge_type,
            "properties": clean_props,
        },
    }


def get_text_content(content: str | list) -> str:
    """Extract text content from message content field."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)
    return ""


def convert_transcript(input_path: Path) -> tuple[list[dict], dict[str, int]]:
    """
    Convert a transcript JSONL file to ManifoldDB backup format.

    Returns:
        Tuple of (records list, statistics dict)
    """
    entities = []
    edges = []

    id_counter = [0]
    edge_id_counter = [0]

    # Maps for linking
    uuid_to_entity_id: dict[str, int] = {}
    tool_use_id_to_entity_id: dict[str, int] = {}

    # First pass: read all records
    records = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Second pass: create entities
    for record in records:
        record_type = record.get("type", "unknown")
        uuid = record.get("uuid")

        # Determine labels based on type
        if record_type == "user":
            labels = ["Message", "User"]
        elif record_type == "assistant":
            labels = ["Message", "Assistant"]
        elif record_type == "queue-operation":
            labels = ["Message", "QueueOperation"]
        else:
            labels = ["Message", record_type.title()]

        # Extract message properties
        message = record.get("message", {})
        content = message.get("content", "")

        # Build properties
        props = {
            "uuid": uuid,
            "sessionId": record.get("sessionId"),
            "timestamp": record.get("timestamp"),
            "messageType": record_type,
            "role": message.get("role"),
            "cwd": record.get("cwd"),
            "gitBranch": record.get("gitBranch"),
            "userType": record.get("userType"),
            "version": record.get("version"),
            "isSidechain": record.get("isSidechain"),
            "parentUuid": record.get("parentUuid"),
        }

        # For assistant messages, add usage stats
        if record_type == "assistant":
            props["requestId"] = record.get("requestId")
            props["model"] = message.get("model")
            props["stopReason"] = message.get("stop_reason")
            usage = message.get("usage", {})
            if usage:
                props["inputTokens"] = usage.get("input_tokens")
                props["outputTokens"] = usage.get("output_tokens")
                props["cacheReadTokens"] = usage.get("cache_read_input_tokens")
                props["cacheCreationTokens"] = usage.get("cache_creation_input_tokens")

        # Extract text content
        text_content = get_text_content(content)
        if text_content:
            # Truncate very long content for the property
            if len(text_content) > 10000:
                props["textContent"] = text_content[:10000] + "...[truncated]"
                props["textContentFull"] = text_content  # Keep full version
            else:
                props["textContent"] = text_content

        # For queue-operation, add operation
        if record_type == "queue-operation":
            props["operation"] = record.get("operation")

        # Create message entity
        entity_id = generate_id(id_counter)
        if uuid:
            uuid_to_entity_id[uuid] = entity_id

        entities.append(create_entity_record(entity_id, labels, props))

        # Extract tool uses and results from content
        if isinstance(content, list):
            for block in content:
                block_type = block.get("type")

                if block_type == "tool_use":
                    tool_id = block.get("id")
                    tool_name = block.get("name")
                    tool_input = block.get("input", {})

                    tool_entity_id = generate_id(id_counter)
                    if tool_id:
                        tool_use_id_to_entity_id[tool_id] = tool_entity_id

                    tool_props = {
                        "toolUseId": tool_id,
                        "toolName": tool_name,
                        "input": tool_input,  # Will be JSON serialized
                    }

                    # Add some common input fields as top-level props for querying
                    if tool_name == "Bash":
                        tool_props["command"] = tool_input.get("command")
                        tool_props["description"] = tool_input.get("description")
                    elif tool_name == "Read":
                        tool_props["filePath"] = tool_input.get("file_path")
                    elif tool_name == "Write":
                        tool_props["filePath"] = tool_input.get("file_path")
                    elif tool_name == "Grep":
                        tool_props["pattern"] = tool_input.get("pattern")
                        tool_props["path"] = tool_input.get("path")
                    elif tool_name == "Glob":
                        tool_props["pattern"] = tool_input.get("pattern")
                        tool_props["path"] = tool_input.get("path")

                    entities.append(
                        create_entity_record(tool_entity_id, ["ToolUse", tool_name], tool_props)
                    )

                    # Create USES_TOOL edge
                    edge_id = generate_id(edge_id_counter)
                    edges.append(
                        create_edge_record(edge_id, entity_id, tool_entity_id, "USES_TOOL")
                    )

                elif block_type == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    result_content = block.get("content", "")

                    result_entity_id = generate_id(id_counter)

                    result_props = {
                        "toolUseId": tool_use_id,
                    }

                    # Handle result content (can be string or array)
                    if isinstance(result_content, str):
                        if len(result_content) > 10000:
                            result_props["content"] = result_content[:10000] + "...[truncated]"
                            result_props["contentFull"] = result_content
                        else:
                            result_props["content"] = result_content
                    else:
                        result_props["content"] = result_content  # Will be JSON serialized

                    entities.append(
                        create_entity_record(result_entity_id, ["ToolResult"], result_props)
                    )

                    # Create HAS_RESULT edge from ToolUse to ToolResult
                    if tool_use_id and tool_use_id in tool_use_id_to_entity_id:
                        edge_id = generate_id(edge_id_counter)
                        edges.append(
                            create_edge_record(
                                edge_id,
                                tool_use_id_to_entity_id[tool_use_id],
                                result_entity_id,
                                "HAS_RESULT",
                            )
                        )

    # Third pass: create FOLLOWS edges based on parentUuid
    for record in records:
        uuid = record.get("uuid")
        parent_uuid = record.get("parentUuid")

        if uuid and parent_uuid:
            child_id = uuid_to_entity_id.get(uuid)
            parent_id = uuid_to_entity_id.get(parent_uuid)

            if child_id and parent_id:
                edge_id = generate_id(edge_id_counter)
                edges.append(create_edge_record(edge_id, child_id, parent_id, "FOLLOWS"))

    # Build final output
    stats = {
        "entities": len(entities),
        "edges": len(edges),
    }

    output = [create_metadata_record(stats)]
    output.extend(entities)
    output.extend(edges)
    output.append({
        "type": "end_of_backup",
        "data": {
            "entity_count": stats["entities"],
            "edge_count": stats["edges"],
            "metadata_count": 0,
            "total_records": stats["entities"] + stats["edges"],
            "uncompressed_size": 0,
        }
    })

    return output, stats


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.jsonl> [output.jsonl]", file=sys.stderr)
        print("\nConverts Claude Code session transcripts to ManifoldDB graph format.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_records, stats = convert_transcript(input_path)

    # Output
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
        with open(output_path, "w") as f:
            for record in output_records:
                f.write(json.dumps(record) + "\n")
        print(f"Converted {input_path.name}:", file=sys.stderr)
        print(f"  Entities: {stats['entities']}", file=sys.stderr)
        print(f"  Edges: {stats['edges']}", file=sys.stderr)
        print(f"  Output: {output_path}", file=sys.stderr)
    else:
        for record in output_records:
            print(json.dumps(record))


if __name__ == "__main__":
    main()

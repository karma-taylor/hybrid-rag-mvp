#!/usr/bin/env python3
"""Split Markdown documents into JSON while preserving their directory tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "my_md_docs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "chunked_docs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively split Markdown files and preserve their directory tree."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a document when its matching JSON output already exists.",
    )
    return parser.parse_args()


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", "。", "！", "？", "，", " "],
    )


def make_chunk_id(source_path: Path, chunk_index: int) -> str:
    value = f"{source_path.as_posix()}:{chunk_index}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:16]


def make_derived_id(source_path: Path, kind: str, index: int) -> str:
    """Stable IDs for table parent/row chunks; legacy paragraph IDs remain unchanged."""
    return hashlib.sha256(f"{source_path.as_posix()}:{kind}:{index}".encode("utf-8")).hexdigest()[:16]


def markdown_tables(content: str) -> list[list[str]]:
    """Return contiguous Markdown tables without splitting any row."""
    tables: list[list[str]] = []
    current: list[str] = []
    for line in content.splitlines():
        if line.lstrip().startswith("|"):
            current.append(line)
        elif current:
            if len(current) >= 2:
                tables.append(current)
            current = []
    if len(current) >= 2:
        tables.append(current)
    return tables


def table_chunks(source_path: Path, content: str, department: str) -> list[dict]:
    """Create parent + row chunks so exact fields and whole-table questions coexist."""
    chunks: list[dict] = []
    for table_index, lines in enumerate(markdown_tables(content), start=1):
        table_id = make_derived_id(source_path, "table", table_index)
        header = lines[0]
        rows = [line for line in lines[2:] if not re.fullmatch(r"\s*\|?[-:| ]+\|?\s*", line)]
        parent_text = "\n".join(lines)
        chunks.append({
            "chunk_id": table_id, "chunk_index": table_index, "chunk_type": "table_parent",
            "table_id": table_id, "parent_id": None, "department": department,
            "source_path": source_path.as_posix(), "content": parent_text,
            # A deterministic placeholder is intentionally factual; deployments may replace it with an LLM summary.
            "table_summary": f"表格包含 {len(rows)} 行，表头：{header}",
        })
        for row_index, row in enumerate(rows, start=1):
            chunks.append({
                "chunk_id": make_derived_id(source_path, f"table-{table_index}-row", row_index),
                "chunk_index": row_index, "chunk_type": "table_row", "table_id": table_id,
                "parent_id": table_id, "department": department, "source_path": source_path.as_posix(),
                "content": f"表头：{header}\n数据行：{row}",
            })
    return chunks


def process_file(
    markdown_path: Path,
    input_root: Path,
    output_root: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> int:
    relative_path = markdown_path.relative_to(input_root)
    output_path = (output_root / relative_path).with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = markdown_path.read_text(encoding="utf-8")
    department_path = relative_path.parent.as_posix()
    department = department_path if department_path != "." else "general_public"
    chunks = [text for text in splitter.split_text(content) if text.strip()]
    document_id = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:16]
    paragraph_chunks = [
        {
            "chunk_id": make_chunk_id(relative_path, index),
            "chunk_index": index,
            "chunk_type": "paragraph",
            "document_id": document_id,
            "department": department,
            "source_path": relative_path.as_posix(),
            "version": 1,
            "content": text,
        }
        for index, text in enumerate(chunks, start=1)
    ]
    derived_chunks = table_chunks(relative_path, content, department)
    for item in derived_chunks:
        item["document_id"] = document_id
        item["version"] = 1

    payload = {
        "source_path": relative_path.as_posix(),
        "department": department,
        "document_id": document_id,
        "chunk_count": len(paragraph_chunks) + len(derived_chunks),
        "chunks": paragraph_chunks + derived_chunks,
    }

    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(output_path)
    return len(paragraph_chunks) + len(derived_chunks)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_root}")

    splitter = build_splitter()
    markdown_files = sorted(
        path for path in input_root.rglob("*.md") if not path.name.startswith(".")
    )
    processed_files = skipped_files = total_chunks = 0

    for markdown_path in markdown_files:
        relative_path = markdown_path.relative_to(input_root)
        output_path = (output_root / relative_path).with_suffix(".json")
        if args.skip_existing and output_path.exists():
            skipped_files += 1
            print(f"SKIP    {relative_path}")
            continue

        chunk_count = process_file(markdown_path, input_root, output_root, splitter)
        processed_files += 1
        total_chunks += chunk_count
        print(f"SUCCESS {relative_path} ({chunk_count} chunks)")

    print("---")
    print(f"Markdown files found: {len(markdown_files)}")
    print(f"Files processed: {processed_files}")
    print(f"Files skipped: {skipped_files}")
    print(f"Chunks generated this run: {total_chunks}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()

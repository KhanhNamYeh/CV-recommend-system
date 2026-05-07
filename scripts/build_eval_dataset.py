import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


QUESTION_PATTERN = re.compile(r"Câu hỏi:\s*(.+)", re.IGNORECASE | re.DOTALL)


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def extract_messages(example: Dict) -> Tuple[str, str]:
    messages = example.get("messages", [])
    user_text = ""
    assistant_text = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content
    if not user_text or not assistant_text:
        raise ValueError("Missing user/assistant message in one sample.")
    return user_text, assistant_text


def extract_query(user_text: str) -> str:
    m = QUESTION_PATTERN.search(user_text)
    if m:
        return m.group(1).strip()
    return user_text.strip()


def infer_split(path: Path) -> str:
    stem = path.stem.lower()
    if "train" in stem:
        return "train"
    if "val" in stem or "valid" in stem:
        return "val"
    if "test" in stem:
        return "test"
    return "unknown"


def build_records(paths: List[Path]) -> List[Dict]:
    records: List[Dict] = []
    uid = 0
    for path in paths:
        split = infer_split(path)
        for sample in read_jsonl(path):
            user_text, assistant_text = extract_messages(sample)
            query = extract_query(user_text)
            records.append(
                {
                    "id": f"{split}_{uid:06d}",
                    "split": split,
                    "query": query,
                    "reference_answer": assistant_text.strip(),
                    "source_user_prompt": user_text.strip(),
                }
            )
            uid += 1
    return records


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_split_files(out_dir: Path, rows: List[Dict]) -> None:
    by_split: Dict[str, List[Dict]] = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)
    for split, items in by_split.items():
        write_jsonl(out_dir / f"query_eval_{split}.jsonl", items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build query-eval dataset from train/val/test JSONL files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["data/train.jsonl", "data/val.jsonl", "data/test.jsonl"],
        help="Input JSONL files.",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Output directory for generated files.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    rows = build_records(input_paths)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "query_eval_all.jsonl", rows)
    write_split_files(out_dir, rows)

    print(f"Built {len(rows)} records.")
    print(f"- {out_dir / 'query_eval_all.jsonl'}")
    for split in ["train", "val", "test", "unknown"]:
        split_path = out_dir / f"query_eval_{split}.jsonl"
        if split_path.exists():
            print(f"- {split_path}")


if __name__ == "__main__":
    main()

"""
embed_jd_bgem3.py
=================
Embed `rag_text` của từng JD trong jd_clean.json bằng BGE-M3
(AITeamVN/Vietnamese_Embedding) và lưu thành 1 file .npz duy nhất.

Chiến lược chunking: 1 JD = 1 chunk = 1 vector (rag_text đã tổng hợp gọn).

Output: jd_embeddings.npz
  - embeddings : float32 ndarray [N, 1024]   (đã L2-normalize → dùng dot = cosine)
  - jd_ids     : object  ndarray [N]         (vd: "JD-0008")
  - texts      : object  ndarray [N]         (rag_text gốc)
  - companies  : object  ndarray [N]
  - titles     : object  ndarray [N]
  - cities     : object  ndarray [N]

Sau này nạp vào FAISS/Chroma:
    data = np.load("jd_embeddings.npz", allow_pickle=True)
    index = faiss.IndexFlatIP(1024); index.add(data["embeddings"])
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

MODEL_ID   = "AITeamVN/Vietnamese_Embedding"
MAX_LENGTH = 512
BATCH_SIZE = 16
EMB_DIM    = 1024


@torch.no_grad()
def encode_batch(texts: list[str], tokenizer, model, device: str) -> np.ndarray:
    """Encode 1 batch text → ndarray [B, 1024], đã L2-normalize."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    out = model(**enc, return_dict=True)
    # BGE-M3 dense embedding = last_hidden_state của [CLS] (token đầu)
    cls = out.last_hidden_state[:, 0]              # [B, 1024]
    cls = F.normalize(cls, p=2, dim=-1)            # cosine-ready
    return cls.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="jd_clean.json")
    parser.add_argument("--output", default="jd_embeddings.npz")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--hf-cache", default="hf_cache")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    print(f"[1/4] Đọc {in_path}")
    jds = json.loads(in_path.read_text(encoding="utf-8"))
    print(f"      {len(jds)} JD")

    # Lọc JD có rag_text hợp lệ
    items = []
    for jd in jds:
        text = (jd.get("rag_text") or "").strip()
        if not text:
            continue
        header = jd.get("header") or {}
        items.append({
            "jd_id":   jd.get("jd_id", ""),
            "text":    text,
            "company": jd.get("company", ""),
            "title":   header.get("title", ""),
            "city":    header.get("city", ""),
        })
    print(f"      Có rag_text: {len(items)}")

    # Use a writable cache directory to avoid read-only filesystem errors.
    hf_cache_dir = Path(args.hf_cache).resolve()
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hf_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Import after setting cache envs to ensure HF picks up writable paths.
    from transformers import AutoModel, AutoTokenizer

    print(f"[2/4] Load model {MODEL_ID} ({args.device})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(hf_cache_dir))
    model = AutoModel.from_pretrained(MODEL_ID, cache_dir=str(hf_cache_dir)).to(args.device).eval()

    print(f"[3/4] Encode {len(items)} chunks, batch={args.batch_size}")
    embs = np.zeros((len(items), EMB_DIM), dtype=np.float32)
    for i in tqdm(range(0, len(items), args.batch_size)):
        batch_texts = [it["text"] for it in items[i : i + args.batch_size]]
        embs[i : i + len(batch_texts)] = encode_batch(
            batch_texts, tokenizer, model, args.device
        )

    print(f"[4/4] Lưu → {out_path}")
    np.savez(
        out_path,
        embeddings = embs,
        jd_ids     = np.array([it["jd_id"]   for it in items], dtype=object),
        texts      = np.array([it["text"]    for it in items], dtype=object),
        companies  = np.array([it["company"] for it in items], dtype=object),
        titles     = np.array([it["title"]   for it in items], dtype=object),
        cities     = np.array([it["city"]    for it in items], dtype=object),
    )
    print(f"OK — shape embeddings = {embs.shape}, file size = "
          f"{out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

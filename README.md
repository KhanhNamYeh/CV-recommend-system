# RAG Tuyển Dụng IT Tiếng Việt

Hệ thống tư vấn tuyển dụng IT sử dụng Retrieval-Augmented Generation (RAG) kết hợp mô hình ngôn ngữ Qwen3-4B fine-tuned trên dữ liệu JD tiếng Việt.

## Kiến trúc hệ thống

Pipeline RAG với 4 cấu hình đánh giá:

| Config | LLM | RAG | Mô tả |
|--------|-----|-----|-------|
| A | Base | Không | Baseline |
| B | Base | Có | Base + hybrid retrieval |
| C | Fine-tuned | Không | Instruction-tuned |
| **D** | **Fine-tuned** | **Có** | **Tốt nhất** |

**Retrieval pipeline:** Dense (Vietnamese_Embedding_v2) + BM25 → RRF fusion → Rerank (BGE-reranker-v2-m3) → Top-5

## Cấu trúc project

```
├── app.py                          # Gradio web UI
├── notebooks/
│   ├── 01_finetune_qwen3_4b.ipynb  # QLoRA fine-tuning (chạy trên Kaggle)
│   ├── 02_retrieval_eval.ipynb     # Đánh giá retrieval pipeline
│   └── 03_rag_eval.ipynb           # Đánh giá 4 config RAG
├── scripts/
│   ├── embed_jd.py                 # Tạo JD embeddings
│   └── build_eval_dataset.py       # Chuẩn bị dữ liệu eval
├── data/
│   ├── train.jsonl / val.jsonl / test.jsonl
│   ├── jd_full_text.json           # 580 job descriptions
│   ├── jd_clean.json               # JD với rag_text (input cho embed_jd.py)
│   ├── qa_dataset_clean.json       # 696 cặp QA
│   └── eval/
│       ├── eval.json               # 847 queries đánh giá
│       └── query_retrieval_eval.json
└── hf_cache/                       # HuggingFace model cache
```

## Cài đặt & Chạy

```bash
pip install -r requirements.txt

# Tạo embeddings (cần GPU)
python scripts/embed_jd.py

# Chạy web UI
python app.py
```

## Kết quả

| Config | BLEU | ROUGE-L | BERTScore |
|--------|------|---------|-----------|
| A (Base) | 5.81 | 34.59 | 89.49 |
| B (Base+RAG) | 12.34 | 43.75 | 91.50 |
| C (FT) | 32.28 | 57.07 | 93.24 |
| **D (FT+RAG)** | **46.95** | **68.26** | **95.67** |

Retrieval Hybrid+Rerank: Hit@1=45.9%, Hit@5=69.4%, Recall@5=51.0%

import gc
import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# HF cache (set before importing HF/Transformers)
# ============================================================
HF_CACHE_DIR = Path("hf_cache")
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE = str(HF_CACHE_DIR)

os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = HF_CACHE

import faiss
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from rank_bm25 import BM25Okapi
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ============================================================
# Paths and config (match eval_4config_rag.ipynb)
# ============================================================
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "qwen3-4b-qlora-jd/best"

DATA_DIR = Path("data")
JD_JSON_PATH = DATA_DIR / "jd_full_text.json"
JD_EMB_PATH = DATA_DIR / "jd_embeddings.npz"
FAISS_INDEX_PATH = DATA_DIR / "jd_faiss.index"

DENSE_MODEL_ID = "AITeamVN/Vietnamese_Embedding_v2"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
TOP_K = 5
DENSE_TOP_K = 50
BM25_TOP_K = 50
FUSION_TOP_K = 50
RRF_K = 60
BATCH_EMBED = 16
MAX_LEN_EMBED = 512
MAX_LEN_RERANK = 1024

MAX_NEW_TOKENS = 256
MAX_NEW_TOKENS_RAG = 128
MAX_INPUT_TOKENS = 1536

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RETRIEVER_DEVICE = os.environ.get("RETRIEVER_DEVICE", "cpu")

# ============================================================
# Helpers
# ============================================================

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def parse_user_msg(user_content: str) -> Tuple[str, Optional[str]]:
    """Split (question, candidate_profile) from user message text."""
    q_match = re.search(r"Câu hỏi:\s*(.+)", user_content, re.DOTALL)
    question = q_match.group(1).strip() if q_match else user_content.strip()

    p_match = re.search(
        r"Hồ sơ ứng viên:\s*(.+?)(?=\n\nCâu hỏi:)",
        user_content,
        re.DOTALL,
    )
    profile = p_match.group(1).strip() if p_match else None
    return question, profile


def load_system_prompt() -> str:
    train_path = DATA_DIR / "train.jsonl"
    if train_path.exists():
        with train_path.open("r", encoding="utf-8") as f:
            sample = json.loads(f.readline())
        return next(m["content"] for m in sample["messages"] if m["role"] == "system")
    return (
        "Trợ lý tư vấn việc làm IT tại Việt Nam. "
        "Trả lời dựa trên thông tin tuyển dụng được cung cấp, chính xác và ngắn gọn. "
        "Không dùng 'em' hay 'bạn' trong câu trả lời — viết trung lập như thông tin tra cứu. "
        "Nếu thông tin không có trong dữ liệu, nói rõ thay vì đoán."
    )


SYSTEM_PROMPT = load_system_prompt()


def load_sample_from_train(seed: Optional[int] = None) -> Tuple[str, str]:
    train_path = DATA_DIR / "train.jsonl"
    if not train_path.exists():
        return "", ""

    rng = random.Random(seed)
    chosen = None
    with train_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            user = next((m for m in obj.get("messages", []) if m.get("role") == "user"), None)
            if not user:
                continue
            question, profile = parse_user_msg(user.get("content", ""))
            if not question or not profile:
                continue
            if chosen is None or rng.randint(0, i) == 0:
                chosen = (question, profile)

    if not chosen:
        return "", ""

    question, profile = chosen
    return question, profile


SAMPLE_QUESTION, SAMPLE_PROFILE = load_sample_from_train()


def build_jd_context(jd_ids: List[str], jd_by_id: Dict[str, dict], doc_texts: List[str], doc_id_to_idx: Dict[str, int]) -> str:
    blocks = []
    for jd_id in jd_ids:
        jd = jd_by_id.get(jd_id)
        if jd:
            text = jd.get("text", "")
        else:
            idx = doc_id_to_idx.get(jd_id)
            text = doc_texts[idx] if idx is not None else ""
        if text:
            blocks.append(f"[{jd_id}]\n{text}")
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    profile: Optional[str],
    jd_ids: Optional[List[str]],
    jd_by_id: Dict[str, dict],
    doc_texts: List[str],
    doc_id_to_idx: Dict[str, int],
) -> List[Dict[str, str]]:
    parts = []
    if jd_ids:
        ctx = build_jd_context(jd_ids, jd_by_id, doc_texts, doc_id_to_idx)
        if ctx:
            parts.append(f"Thông tin tuyển dụng:\n{ctx}")
    if profile:
        parts.append(f"Hồ sơ ứng viên: {profile}")
    parts.append(f"Câu hỏi: {question}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# ============================================================
# Load JD data + retrieval stack
# ============================================================

with JD_JSON_PATH.open("r", encoding="utf-8") as f:
    jd_raw = json.load(f)

jd_by_id: Dict[str, dict] = {}
for jd in (jd_raw if isinstance(jd_raw, list) else []):
    jid = str(jd.get("jd_id", "")).strip()
    if jid:
        jd_by_id[jid] = jd

if JD_EMB_PATH.exists():
    emb_npz = np.load(JD_EMB_PATH, allow_pickle=True)
    doc_ids = [str(x) for x in emb_npz["jd_ids"]]
    doc_texts = [_norm(str(x)) for x in emb_npz["texts"]]
else:
    doc_ids = []
    doc_texts = []
    for jd in (jd_raw if isinstance(jd_raw, list) else []):
        jid = str(jd.get("jd_id", "")).strip()
        if not jid:
            continue
        doc_ids.append(jid)
        doc_texts.append(_norm(str(jd.get("text", ""))))

doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

try:
    from underthesea import word_tokenize as _uts

    def vi_tokenize(text: str) -> List[str]:
        return [t for t in _uts(text, format="text").split() if t]

except Exception:
    try:
        from pyvi import ViTokenizer as _vt

        def vi_tokenize(text: str) -> List[str]:
            return [t.replace("_", " ") for t in _vt.tokenize(text).split() if t]

    except Exception:
        _RE = re.compile(r"\w+", re.UNICODE)

        def vi_tokenize(text: str) -> List[str]:
            return _RE.findall(text.lower())


def bm25_tokenize(text: str) -> List[str]:
    return vi_tokenize(_norm(text).lower())


bm25_index = BM25Okapi([bm25_tokenize(t) for t in doc_texts])

from transformers import AutoTokenizer as _ATokenizer

dense_tokenizer = _ATokenizer.from_pretrained(DENSE_MODEL_ID, cache_dir=HF_CACHE)
dense_encoder = AutoModel.from_pretrained(DENSE_MODEL_ID, cache_dir=HF_CACHE)

def _safe_to_device(model, device_name: str) -> Tuple[torch.nn.Module, str]:
    try:
        return model.to(device_name), device_name
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model.to("cpu"), "cpu"


dense_encoder, RETRIEVER_DEVICE = _safe_to_device(dense_encoder, RETRIEVER_DEVICE)
dense_encoder.eval()

rerank_tokenizer = _ATokenizer.from_pretrained(RERANKER_MODEL_ID, cache_dir=HF_CACHE)
rerank_model = AutoModelForSequenceClassification.from_pretrained(
    RERANKER_MODEL_ID, cache_dir=HF_CACHE
)
rerank_model, RETRIEVER_DEVICE = _safe_to_device(rerank_model, RETRIEVER_DEVICE)
rerank_model.eval()

def encode_docs(texts: List[str]) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(texts), BATCH_EMBED):
        batch = [_norm(t) for t in texts[i : i + BATCH_EMBED]]
        enc = dense_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN_EMBED,
            return_tensors="pt",
        ).to(RETRIEVER_DEVICE)
        with torch.inference_mode():
            out = dense_encoder(**enc, return_dict=True)
            cls = F.normalize(out.last_hidden_state[:, 0], p=2, dim=-1)
        all_vecs.append(cls.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(all_vecs, axis=0)


if JD_EMB_PATH.exists():
    doc_embs = emb_npz["embeddings"].astype(np.float32)
    doc_embs /= np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12
else:
    doc_embs = encode_docs(doc_texts)
    np.savez(JD_EMB_PATH, embeddings=doc_embs, jd_ids=np.array(doc_ids), texts=np.array(doc_texts))

if FAISS_INDEX_PATH.exists():
    faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    if faiss_index.ntotal != doc_embs.shape[0] or faiss_index.d != doc_embs.shape[1]:
        faiss_index = faiss.IndexFlatIP(doc_embs.shape[1])
        faiss_index.add(doc_embs)
        faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
else:
    faiss_index = faiss.IndexFlatIP(doc_embs.shape[1])
    faiss_index.add(doc_embs)
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))


def encode_queries(texts: List[str]) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(texts), BATCH_EMBED):
        batch = [_norm(t) for t in texts[i : i + BATCH_EMBED]]
        enc = dense_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN_EMBED,
            return_tensors="pt",
        ).to(RETRIEVER_DEVICE)
        with torch.inference_mode():
            out = dense_encoder(**enc, return_dict=True)
            cls = F.normalize(out.last_hidden_state[:, 0], p=2, dim=-1)
        all_vecs.append(cls.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(all_vecs, axis=0)


def dense_search(query: str, top_k: int = DENSE_TOP_K) -> List[Tuple[str, float, int]]:
    q = encode_queries([query])
    scores, idx = faiss_index.search(q, top_k)
    return [(doc_ids[i], float(scores[0][r]), r + 1) for r, i in enumerate(idx[0])]


def bm25_search(query: str, top_k: int = BM25_TOP_K) -> List[Tuple[str, float, int]]:
    toks = bm25_tokenize(query)
    scores = bm25_index.get_scores(toks)
    idx = np.argpartition(-scores, top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    return [(doc_ids[i], float(scores[i]), rank + 1) for rank, i in enumerate(idx)]


def rrf_fusion(result_lists: List[List[Tuple[str, float, int]]], k: int = RRF_K) -> List[Tuple[str, float]]:
    fused: Dict[str, float] = {}
    for results in result_lists:
        for doc_id, _, rank in results:
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


@torch.no_grad()
def rerank(query: str, candidate_ids: List[str], top_k: int = TOP_K) -> List[str]:
    query = _norm(query)
    valid = [d for d in candidate_ids if d in doc_id_to_idx]
    pairs = [[query, doc_texts[doc_id_to_idx[d]]] for d in valid]
    enc = rerank_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=MAX_LEN_RERANK,
        return_tensors="pt",
    ).to(RETRIEVER_DEVICE)
    scores = rerank_model(**enc, return_dict=True).logits.view(-1).cpu().numpy()
    return [valid[i] for i in np.argsort(-scores)][:top_k]


def hybrid_rerank(query: str, top_k: int = TOP_K) -> List[str]:
    fused = rrf_fusion(
        [
            dense_search(query, top_k=DENSE_TOP_K),
            bm25_search(query, top_k=BM25_TOP_K),
        ]
    )
    candidates = [d for d, _ in fused[:FUSION_TOP_K]]
    return rerank(query, candidates, top_k=top_k)


def format_top_chunks(jd_ids: List[str], max_chars: int = 360) -> str:
    lines = []
    for rank, jd_id in enumerate(jd_ids, start=1):
        jd = jd_by_id.get(jd_id, {})
        title = jd.get("title", "")
        company = jd.get("company", "")
        city = jd.get("city", "")
        text = jd.get("text", "")
        if not text:
            idx = doc_id_to_idx.get(jd_id)
            text = doc_texts[idx] if idx is not None else ""
        snippet = text.replace("\n", " ")[:max_chars]
        meta = " | ".join([x for x in [title, company, city] if x])
        if meta:
            lines.append(f"{rank}. [{jd_id}] {meta}\n   {snippet}")
        else:
            lines.append(f"{rank}. [{jd_id}] {snippet}")
    return "\n\n".join(lines) if lines else "(no chunks)"


# ============================================================
# LLM loading
# ============================================================


def _bnb_config(enable_cpu_offload: bool = False) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=enable_cpu_offload,
    )


def _max_memory() -> Dict:
    if torch.cuda.is_available():
        total_gb = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
        gpu_gb = max(total_gb - 1, 1)
        return {0: f"{gpu_gb}GiB", "cpu": "48GiB"}
    return {"cpu": "48GiB"}


def _load_base_model_only():
    try:
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=_bnb_config(False),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=HF_CACHE,
        )
    except ValueError as e:
        if "llm_int8_enable_fp32_cpu_offload" not in str(e):
            raise
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=_bnb_config(True),
            device_map="auto",
            max_memory=_max_memory(),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=HF_CACHE,
        )


def load_base_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    mdl = _load_base_model_only()
    mdl.config.use_cache = True
    return tok, mdl


def load_ft_model():
    if not Path(ADAPTER_PATH).exists():
        raise FileNotFoundError(
            f"Adapter not found: {Path(ADAPTER_PATH).resolve()}"
        )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = _load_base_model_only()
    ft = PeftModel.from_pretrained(base, ADAPTER_PATH)
    ft.config.use_cache = True
    return tok, ft


def free_model(*models):
    for m in models:
        del m
    gc.collect()
    torch.cuda.empty_cache()


_MODEL_STATE = {"kind": None, "tokenizer": None, "model": None}


def get_model(kind: str):
    if _MODEL_STATE["kind"] == kind and _MODEL_STATE["model"] is not None:
        return _MODEL_STATE["tokenizer"], _MODEL_STATE["model"]

    if _MODEL_STATE["model"] is not None:
        free_model(_MODEL_STATE["model"])

    if kind == "base":
        tok, mdl = load_base_model()
    else:
        tok, mdl = load_ft_model()

    _MODEL_STATE.update({"kind": kind, "tokenizer": tok, "model": mdl})
    return tok, mdl


# ============================================================
# Chat function
# ============================================================


def infer_answer(question: str, profile: str, config: str) -> Tuple[str, str]:
    use_rag = config in {"B", "D"}
    model_kind = "base" if config in {"A", "B"} else "ft"

    jd_ids = hybrid_rerank(question, top_k=TOP_K)
    chunks_md = format_top_chunks(jd_ids)

    tok, mdl = get_model(model_kind)
    max_new = MAX_NEW_TOKENS_RAG if use_rag else MAX_NEW_TOKENS
    max_inp = MAX_INPUT_TOKENS if use_rag else None
    use_cache = False if use_rag else True

    prompt = build_messages(question, profile, jd_ids if use_rag else None, jd_by_id, doc_texts, doc_id_to_idx)

    ct_kwargs = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if max_inp is not None:
        ct_kwargs["truncation"] = True
        ct_kwargs["max_length"] = max_inp

    with torch.inference_mode():
        inputs = tok.apply_chat_template(prompt, **ct_kwargs).to(mdl.device)
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            use_cache=use_cache,
        )
        text = tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()

    return text, chunks_md


def _history_to_messages(history) -> List[Dict[str, str]]:
    if not history:
        return []
    if isinstance(history, list) and history and isinstance(history[0], dict):
        return history
    messages = []
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user, bot = item
            if user:
                messages.append({"role": "user", "content": str(user)})
            if bot:
                messages.append({"role": "assistant", "content": str(bot)})
    return messages


def chat_submit(user_text: str, history, model_choice: str, cv_text: str):
    if not user_text:
        return history, "", ""

    answer, chunks_md = infer_answer(user_text, cv_text, model_choice)
    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": answer})
    return messages, chunks_md, ""


def clear_chat():
    return [], "", ""


def sample_refresh():
    question, profile = load_sample_from_train()
    return profile, question, [], ""


# ============================================================
# UI
# ============================================================

with gr.Blocks(title="LLM x RAG Demo") as demo:
    gr.Markdown("# LLM x RAG Demo (A/B/C/D)")
    gr.Markdown(
        "Chon cau hinh va nhap CV. Khi hoi dap, he thong se hien top-5 chunks tu RAG."
    )

    with gr.Row():
        model_choice = gr.Dropdown(
            label="Model config",
            choices=[
                ("A - Base, No RAG", "A"),
                ("B - Base, RAG", "B"),
                ("C - FT, No RAG", "C"),
                ("D - FT, RAG", "D"),
            ],
            value="B",
        )
        cv_text = gr.Textbox(
            label="CV / Profile",
            placeholder="Nhap CV / gioi thieu (tuy chon)",
            lines=6,
            value=SAMPLE_PROFILE,
        )

    chatbot = gr.Chatbot(label="Chat", height=360)
    user_text = gr.Textbox(
        label="Question",
        placeholder="Nhap cau hoi...",
        value=SAMPLE_QUESTION,
    )

    with gr.Row():
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear")
        sample_btn = gr.Button("Random sample")

    chunks_box = gr.Textbox(
        label="Top-5 chunks", value="", lines=12, interactive=False
    )

    send_btn.click(
        fn=chat_submit,
        inputs=[user_text, chatbot, model_choice, cv_text],
        outputs=[chatbot, chunks_box, user_text],
    )
    user_text.submit(
        fn=chat_submit,
        inputs=[user_text, chatbot, model_choice, cv_text],
        outputs=[chatbot, chunks_box, user_text],
    )
    clear_btn.click(fn=clear_chat, outputs=[chatbot, chunks_box, user_text])
    sample_btn.click(fn=sample_refresh, outputs=[cv_text, user_text, chatbot, chunks_box])


demo.launch(server_name="127.0.0.1", server_port=7860)

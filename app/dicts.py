import json
from pathlib import Path
from typing import List
import numpy as np

def safe_cosine_sims(m: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    m_norm = np.linalg.norm(m, axis=1, keepdims=True)
    q_norm = np.linalg.norm(q)

    m_norm = np.clip(m_norm, 1e-12, None)
    q_norm = max(float(q_norm), 1e-12)

    m = m / m_norm
    q = q / q_norm

    sims = safe_cosine_sims(m, q)
    sims = np.nan_to_num(sims, nan=-1e9, posinf=-1e9, neginf=-1e9)
    return sims

from .models import DictItem

def load_dict(path: str) -> List[DictItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DictItem(**x) for x in data]

def dict_text_for_embedding(item: DictItem) -> str:
    parts = [item.id, item.name]
    if item.description_short:
        parts.append(item.description_short)
    if item.risk_type:
        parts.append(f"категория риска: {item.risk_type}")
    if item.scenario_short:
        parts.append(item.scenario_short)
    return " | ".join([p for p in parts if p])

def cosine_topk(query_vec: np.ndarray, mat: np.ndarray, k: int) -> tuple[List[int], List[float]]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    sims = m @ q
    idx = np.argsort(-sims)[:k]
    scores = sims[idx]
    return idx.tolist(), scores.tolist()

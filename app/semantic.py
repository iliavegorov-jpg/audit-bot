import numpy as np
from typing import List, Dict
from .models import DictItem, UserInput
from .dicts import dict_text_for_embedding, cosine_topk

def build_user_input_text(ui: UserInput) -> str:
    parts = [
        f"проблема: {ui.problem_text}",
        f"процесс/объект: {ui.process_object}",
        f"период: {ui.period}",
    ]
    if ui.participants_roles: parts.append(f"участники (роли): {ui.participants_roles}")
    if ui.what_violated: parts.append(f"что нарушено: {ui.what_violated}")
    if ui.amounts_terms: parts.append(f"суммы/сроки: {ui.amounts_terms}")
    if ui.documents: parts.append(f"документы: {ui.documents}")
    return "\n".join(parts)

def precompute_embeddings(llm, items: List[DictItem]) -> np.ndarray:
    vecs = []
    for it in items:
        vecs.append(llm.embed(dict_text_for_embedding(it)))
    return np.vstack(vecs)

def make_candidates(items: List[DictItem], idxs: List[int], scores: List[float]) -> List[dict]:
    out = []
    for i, score in zip(idxs, scores):
        it = items[i]
        d = {"id": it.id, "name": it.name, "confidence": round(float(score) * 100, 1)}
        if it.description_short:
            d["description_short"] = it.description_short
        if it.risk_type:
            d["risk_type"] = it.risk_type
        if it.scenario_short:
            d["scenario_short"] = it.scenario_short
        out.append(d)
    return out

def topk_candidates(llm, ui: UserInput, categories: List[DictItem], risks: List[DictItem],
                    cat_mat: np.ndarray, risk_mat: np.ndarray, k: int = 20) -> Dict[str, List[dict]]:
    q_text = build_user_input_text(ui)
    q_vec = llm.embed(q_text)

    cat_idx, cat_scores = cosine_topk(q_vec, cat_mat, min(k, len(categories)))
    risk_idx, risk_scores = cosine_topk(q_vec, risk_mat, min(k, len(risks)))

    return {
        "deviation_categories": make_candidates(categories, cat_idx, cat_scores),
        "risks": make_candidates(risks, risk_idx, risk_scores),
    }


from pathlib import Path

def load_cached_matrices(cat_path: str = "./emb_cache/cat_mat.npy", risk_path: str = "./emb_cache/risk_mat.npy"):
    cat_file = Path(cat_path)
    risk_file = Path(risk_path)
    if not cat_file.exists() or not risk_file.exists():
        return None, None
    return np.load(cat_file), np.load(risk_file)

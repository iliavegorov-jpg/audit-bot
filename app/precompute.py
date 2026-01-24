import numpy as np
from pathlib import Path
from .config import get_settings
from .dicts import load_dict, dict_text_for_embedding
from .yandex_llm import YandexLLM

SET = get_settings()

def main():
    categories = load_dict("./data/deviation_categories.json")
    risks = load_dict("./data/risks.json")

    llm = YandexLLM(
        api_key=SET.yc_api_key,
        completion_model_uri=SET.yc_completion_model_uri,
        embedding_model_uri=SET.yc_embedding_model_uri,
        embedding_dim=SET.yc_embedding_dim,
    )

    out_dir = Path("./emb_cache")
    out_dir.mkdir(exist_ok=True)

    def compute(items, out_path: Path):
        vecs = []
        total = len(items)
        for i, it in enumerate(items, 1):
            text = dict_text_for_embedding(it)
            v = llm.embed(text)
            vecs.append(v)
            if i % 10 == 0 or i == total:
                print(f"{out_path.name}: {i}/{total}")
        mat = np.vstack(vecs)
        np.save(out_path, mat)
        print(f"saved: {out_path} shape={mat.shape}")

    compute(categories, out_dir / "cat_mat.npy")
    compute(risks, out_dir / "risk_mat.npy")

if __name__ == "__main__":
    main()

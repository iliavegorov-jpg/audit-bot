#!/usr/bin/env python3
"""
тестовый скрипт для проверки yandexgpt с новым промптом
"""
import json
import sys
sys.path.insert(0, '/tmp')

from app.config import get_settings
from app.yandex_llm import YandexLLM
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.dicts import load_dict
from app.semantic import topk_candidates, load_cached_matrices, precompute_embeddings
from app.models import UserInput, LLMResponse

def test_llm_completion():
    print("=== ТЕСТ YANDEXGPT С НОВЫМ ПРОМПТОМ ===\n")
    
    # загружаем конфиг
    SET = get_settings()
    
    # инициализируем llm
    llm = YandexLLM(
        api_key=SET.yc_api_key,
        completion_model_uri=SET.yc_completion_model_uri,
        embedding_model_uri=SET.yc_embedding_model_uri,
        embedding_dim=SET.yc_embedding_dim,
    )
    
    # загружаем справочники
    CATEGORIES = load_dict("./data/deviation_categories.json")
    RISKS = load_dict("./data/risks.json")
    
    print(f"✓ загружено категорий: {len(CATEGORIES)}")
    print(f"✓ загружено рисков: {len(RISKS)}\n")
    
    # пробуем загрузить кэш или создать эмбеддинги
    CAT_MAT, RISK_MAT = load_cached_matrices()
    if CAT_MAT is None or RISK_MAT is None:
        print("⚠ нет кэша эмбеддингов, создаём...")
        CAT_MAT = precompute_embeddings(llm, CATEGORIES)
        RISK_MAT = precompute_embeddings(llm, RISKS)
        print("✓ эмбеддинги созданы\n")
    else:
        print("✓ эмбеддинги загружены из кэша\n")
    
    # тестовое отклонение
    ui = {
        "problem_text": "не оформлен договор с подрядчиком на выполнение работ по ремонту",
        "process_object": "управление договорами",
        "period": "3кв 2024",
        "participants_roles": "отдел закупок, подрядчик ООО Строймонтаж",
        "what_violated": "положение о договорной работе, 44-фз",
        "amounts_terms": "работы выполнены на 2.5 млн руб",
        "documents": "акты выполненных работ, счёт на оплату"
    }
    
    ui_obj = UserInput(**ui)
    
    print("тестовое отклонение:")
    print(f"  проблема: {ui['problem_text']}")
    print(f"  процесс: {ui['process_object']}")
    print(f"  период: {ui['period']}\n")
    
    # семантический поиск кандидатов
    print("запуск семантического поиска...")
    candidates = topk_candidates(llm, ui_obj, CATEGORIES, RISKS, CAT_MAT, RISK_MAT, k=20)
    print(f"✓ найдено кандидатов категорий: {len(candidates.get('deviation_categories', []))}")
    print(f"✓ найдено кандидатов рисков: {len(candidates.get('risks', []))}\n")
    
    # формируем промпт
    messages = [
        {"role": "system", "text": SYSTEM_PROMPT},
        {"role": "user", "text": build_user_prompt(ui, candidates)},
    ]
    
    print("отправка запроса в yandexgpt...")
    print("system prompt длина:", len(SYSTEM_PROMPT), "символов")
    print("user prompt длина:", len(messages[1]["text"]), "символов\n")
    
    # делаем запрос
    try:
        raw = llm.completion(messages, temperature=0.2, max_tokens=3800)
        print("✓ получен ответ от yandexgpt")
        print(f"  длина ответа: {len(raw)} символов\n")
        
        # чистим ответ от markdown
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else ""
            if "```" in clean:
                clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()
        
        # парсим json
        print("парсинг json...")
        data = json.loads(clean)
        
        # валидация через pydantic
        print("валидация через pydantic...")
        parsed = LLMResponse(**data)
        
        print("\n=== РЕЗУЛЬТАТ ===")
        print(f"✓ категория отклонения: {parsed.selected['deviation_category'].primary_id}")
        print(f"  confidence: {parsed.selected['deviation_category'].confidence}")
        print(f"  rationale: {parsed.selected['deviation_category'].rationale[:100]}...")
        
        print(f"\n✓ риск: {parsed.selected['risk'].primary_id}")
        print(f"  confidence: {parsed.selected['risk'].confidence}")
        print(f"  rationale: {parsed.selected['risk'].rationale[:100]}...")
        
        print(f"\n✓ секций сгенерировано: {len(parsed.sections)}")
        for section_key, section_data in parsed.sections.items():
            variant_count = len(section_data.variants)
            print(f"  {section_key}: {variant_count} вариантов")
            if variant_count > 0:
                first_var = section_data.variants[0]
                print(f"    v1 short: {first_var.short[:80]}...")
        
        print("\n=== ТЕСТ УСПЕШЕН ===")
        return True
        
    except json.JSONDecodeError as e:
        print(f"\n✗ ОШИБКА ПАРСИНГА JSON: {e}")
        print(f"\nсырой ответ (первые 500 символов):\n{raw[:500]}")
        print(f"\nочищенный ответ (первые 500 символов):\n{clean[:500]}")
        return False
        
    except Exception as e:
        print(f"\n✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_completion()
    sys.exit(0 if success else 1)

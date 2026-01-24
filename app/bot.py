import json
_loads = lambda s: json.loads(s) if s else {}
import json
_loads = lambda s: json.loads(s) if s else {}
import os
from typing import Dict, Any


def to_jsonable(obj):
    # recursively convert pydantic models / dataclasses to plain jsonable types
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.utils.keyboard import InlineKeyboardBuilder



def cat_title(cat_id: str) -> str:
    for c in CATEGORIES:
        obj = c.model_dump() if hasattr(c, 'model_dump') else c
        if isinstance(obj, dict):
            cid = obj.get('id') or obj.get('code')
            if cid == cat_id:
                return obj.get('title') or obj.get('name') or cat_id
    return cat_id

def risk_title(risk_id: str) -> str:
    for r in RISKS:
        obj = r.model_dump() if hasattr(r, 'model_dump') else r
        if isinstance(obj, dict):
            rid = obj.get('id') or obj.get('code')
            if rid == risk_id:
                return obj.get('title') or obj.get('name') or risk_id
    return risk_id
from .config import get_settings
from .db import (
    connect, init_db, create_deviation, get_deviation, update_deviation,
    get_chosen_variant, set_chosen_variant, get_view_mode, toggle_view_mode
)
from .dicts import load_dict
from .yandex_llm import YandexLLM
from .openrouter_llm import OpenRouterLLM
from .semantic import topk_candidates, load_cached_matrices
from .models import UserInput, LLMResponse
from .prompts import SYSTEM_PROMPT, build_user_prompt, build_regen_prompt
from .exporter import export_docx

# Авторизация
VALID_PASSWORD = "audit_2026"

SET = get_settings()

con = connect(SET.sqlite_path)
init_db(con)

# Создаём таблицу для авторизованных пользователей
con.execute("""
CREATE TABLE IF NOT EXISTS authorized_users (
    user_id INTEGER PRIMARY KEY,
    authorized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
con.commit()

def is_authorized(user_id: int) -> bool:
    """Проверка авторизации пользователя"""
    cur = con.execute("SELECT 1 FROM authorized_users WHERE user_id = ?", (user_id,))
    return cur.fetchone() is not None

def authorize_user(user_id: int):
    """Добавление пользователя в список авторизованных"""
    con.execute("INSERT OR IGNORE INTO authorized_users (user_id) VALUES (?)", (user_id,))
    con.commit()

bot = Bot(token=SET.bot_token)
dp = Dispatcher()

class AuthState(StatesGroup):
    waiting_password = State()

class NewDeviation(StatesGroup):
    full_description = State()

class CustomVariant(StatesGroup):
    text_input = State()
    dev_id = State()
    section_key = State()

# dictionaries
CATEGORIES = load_dict("./data/deviation_categories.json")
RISKS = load_dict("./data/risks.json")

llm_embeddings = YandexLLM(
    api_key=SET.yc_api_key,
    completion_model_uri="",  # не используется
    embedding_model_uri=SET.yc_embedding_model_uri,
    embedding_dim=SET.yc_embedding_dim,
)

llm = OpenRouterLLM(api_key=SET.openrouter_api_key, embedding_dim=256)
CAT_MAT, RISK_MAT = load_cached_matrices()
if CAT_MAT is None or RISK_MAT is None:
    print("[WARN] нет кэша эмбеддингов. запусти: python -m app.precompute (один раз), потом перезапусти бота")
    # fallback: медленно посчитаем при старте, но лучше не надо
    from .semantic import precompute_embeddings
    CAT_MAT = precompute_embeddings(llm_embeddings, CATEGORIES)
    RISK_MAT = precompute_embeddings(llm_embeddings, RISKS)

SECTION_ORDER = [
    "essence",
    "root_causes",
    "cost_impact",
    "formulas",
    "risk_cost_scenarios",
    "risk_factors",
    "rsbu_checks",
    "ifrs_checks",
    "measures",
    "export",
]

SECTION_TITLES = {
    "essence": "📝 формулировки отклонения",
    "root_causes": "🔍 коренные причины",
    "cost_impact": "💰 стоимость и влияние на показатели",
    "formulas": "📐 формулы EBITDA и ССДП",
    "risk_cost_scenarios": "📊 стоимость риска (сценарии)",
    "risk_factors": "⚡ риск-факторы",
    "rsbu_checks": "📒 РСБУ: проводки и проверки",
    "ifrs_checks": "📘 МСФО: корректировки",
    "measures": "✅ корректирующие меры",
    "export": "📄 экспорт отчёта",
}

def kb_sections(dev_id: int) -> InlineKeyboardBuilder:
    kb = InlineKeyboardBuilder()
    for key in SECTION_ORDER:
        kb.button(text=SECTION_TITLES[key], callback_data=f"sec|{dev_id}|{key}")
    kb.adjust(1)
    return kb

def kb_section_controls(dev_id: int, section_key: str, current_idx: int, mode: str):
    """кнопка назад"""
    kb = InlineKeyboardBuilder()
    
    # только назад
    kb.button(text="← назад", callback_data=f"back|{dev_id}")
    
    return kb.as_markup()

def render_section(row: Dict[str, Any], section_key: str) -> str:
    sections = _loads(row.get("sections_json"))
    if section_key not in sections:
        return "нет данных по разделу (сначала /build)"
    
    # новая структура: sections[key] содержит {text: "..."}
    section_data = sections[section_key]
    if isinstance(section_data, dict) and "text" in section_data:
        text = section_data["text"]
    elif isinstance(section_data, str):
        text = section_data
    else:
        return "неверный формат данных раздела"
    
    title = SECTION_TITLES.get(section_key, section_key)
    result = f"<b>{title.upper()}</b>\n\n{text}"
    
    # telegram limit 4096 chars, обрезаем если больше
    if len(result) > 4000:
        result = result[:3950] + "\n\n<i>... (текст обрезан, слишком длинный)</i>"
    
    return result

@dp.message(Command("start"))
async def start(m: Message, state: FSMContext):
    user_id = m.from_user.id
    
    # Если уже авторизован в БД, показываем меню
    if is_authorized(user_id):
        await m.answer(
            "⚠️ ВНИМАНИЕ - ОТКАЗ ОТ ОТВЕТСТВЕННОСТИ:\n\n"
            "Данный бот НЕ предназначен для обработки информации, содержащей:\n"
            "• Государственную тайну (ФЗ-5487-1 «О государственной тайне»)\n"
            "• Коммерческую тайну (ФЗ-98 «О коммерческой тайне»)\n"
            "• Персональные данные (ФЗ-152 «О персональных данных»)\n"
            "• Инсайдерскую информацию (ФЗ-224 «О противодействии неправомерному использованию инсайдерской информации»)\n"
            "• Служебную информацию ограниченного распространения\n\n"
            "Вся ответственность за использование конфиденциальной информации, включая:\n"
            "• Разглашение сведений, составляющих гостайну (ст. 283, 283.1 УК РФ)\n"
            "• Нарушение требований по защите персданных (ст. 13.11 КоАП РФ, ст. 137 УК РФ)\n"
            "• Нарушение режима коммерческой тайны (ст. 14.33 КоАП РФ, ст. 183 УК РФ)\n"
            "лежит ИСКЛЮЧИТЕЛЬНО на пользователе в соответствии с:\n"
            "• Федеральным законодательством РФ\n"
            "• Локальными нормативными актами (ЛНА) организации\n"
            "• Должностными инструкциями\n"
            "• Трудовым договором\n\n"
            "Используя бот, вы подтверждаете, что:\n"
            "✓ Ознакомлены с данным предупреждением\n"
            "✓ Несёте полную ответственность за вводимую информацию\n"
            "✓ Не будете вводить конфиденциальные данные\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Команды:\n"
            "/new — новая карточка отклонения\n"
            "/build <id> — сгенерировать анализ\n"
            "/preview <id> — просмотр результатов"
        )
        return
    
    # Если не авторизован, просим пароль
    await state.set_state(AuthState.waiting_password)
    await m.answer(
        "🔐 Для доступа к боту введите пароль:"
    )

@dp.message(AuthState.waiting_password)
async def check_password(m: Message, state: FSMContext):
    if m.text.strip() == VALID_PASSWORD:
        authorize_user(m.from_user.id)
        await state.clear()
        await m.answer("✅ Доступ разрешён! Используйте /start для начала работы")
    else:
        await m.answer("❌ Неверный пароль. Попробуйте ещё раз:")

@dp.message(Command("new"))
async def new(m: Message, state: FSMContext):
    # Проверка авторизации
    if not is_authorized(m.from_user.id):
        await m.answer("❌ Сначала авторизуйтесь через /start")
        return
    
    await state.set_state(NewDeviation.full_description)
    await m.answer(
        "Опиши отклонение по шаблону (если чего-то не знаешь – нормально, я подставлю типовые примеры из практики):\n\n"
        "1️⃣ ЧТО нарушено? (какая норма/договор/лна)\n"
        "2️⃣ ГДЕ? (подразделение/процесс)\n"
        "3️⃣ КОГДА? (дата/период)\n"
        "4️⃣ ПОЧЕМУ? (причина)\n"
        "5️⃣ КТО? (фио + должность ответственного)\n\n"
        "Пример:\n"
        "\"Нарушен п.2.3 Договора №456 от 01.09.2024: произведена оплата без акта приёмки "
        "в отделе снабжения 15.12.2024 из-за отсутствия контроля. "
        "Ответственный: Иванов И.И., нач.отдела снабжения\"\n\n"
        "Можешь писать свободным текстом – главное чтобы эти 5 пунктов были понятны."
    )

@dp.message(NewDeviation.full_description)
async def handle_full_description(m: Message, state: FSMContext):
    # Проверка авторизации
    if not is_authorized(m.from_user.id):
        await state.clear()
        await m.answer("❌ Сначала авторизуйтесь через /start")
        return
    
    description = m.text.strip()
    
    # создаём user_input с единым текстом
    ui = UserInput(
        problem_text=description,
        process_object="",
        period="",
        participants_roles="",
        what_violated="",
        amounts_terms="",
        documents=""
    ).model_dump()
    
    dev_id = create_deviation(con, telegram_user_id=m.from_user.id, user_input=ui)
    await state.clear()
    await m.answer(f"Черновик создан: id={dev_id}\n/build {dev_id} — сгенерировать отчёт")

@dp.message(Command("build"))
async def build(m: Message):
    # Проверка авторизации
    if not is_authorized(m.from_user.id):
        await m.answer("❌ Сначала авторизуйтесь через /start")
        return
    
    parts = m.text.split()
    if len(parts) < 2:
        await m.answer("используй: /build <id>")
        return
    dev_id = int(parts[1])
    row = get_deviation(con, dev_id)
    ui = json.loads(row["user_input_json"])
    ui_obj = UserInput(**ui)

    await m.answer(
        "Генерация структурированного ответа в логике причинно-следственной связи и влияния на бизнес "
        "(powered by Claude Anthropic AI Sonnet 4.5). Это займёт 3-4 минуты."
    )

    candidates = topk_candidates(llm_embeddings, ui_obj, CATEGORIES, RISKS, CAT_MAT, RISK_MAT, k=20)

    messages = [
        {"role": "system", "text": SYSTEM_PROMPT},
        {"role": "user", "text": build_user_prompt(ui, candidates)},
    ]
    raw = llm.completion(messages, temperature=0.2, max_tokens=16000)

    # --- clean model output (remove ```json fences etc.) ---
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else ""
        if "```" in clean:
            clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()
    
    # fix control characters in json strings
    import re
    clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', clean)
    # ------------------------------------------------------

    try:
        data = json.loads(clean)
        
        # PATCH: add missing variants for pydantic validation
        if "sections" in data:
            for section_key, section_data in data["sections"].items():
                if isinstance(section_data, dict) and "text" in section_data and "variants" not in section_data:
                    section_data["variants"] = ["v1", "v2", "v3"]
        
        parsed = LLMResponse(**data)
    except Exception as e:
        await m.answer(f"gpt вернул невалидный json. ошибка: {e}\nсырец (первые 1200 символов):\n{raw[:1200]}")
        return

    sections_dump = {k: v.model_dump() for k, v in parsed.sections.items()}
    update_deviation(con, dev_id, selected=to_jsonable(parsed.selected), sections=sections_dump)
    await m.answer(f"Готово!\n/preview {dev_id} — просмотр и выбор вариантов разделов")

@dp.message(Command("preview"))
async def preview(m: Message):
    # Проверка авторизации
    if not is_authorized(m.from_user.id):
        await m.answer("❌ Сначала авторизуйтесь через /start")
        return
    
    parts = m.text.split()
    if len(parts) < 2:
        await m.answer("используй: /preview <id>")
        return
    dev_id = int(parts[1])
    row = get_deviation(con, dev_id)
    if not row.get("sections_json"):
        await m.answer("нет генерации. сначала /build")
        return

    selected = _loads(row.get("selected_json"))
    
    # показываем классификацию ПЕРВОЙ - 2 категории + 2 риска с обоснованием
    classification_text = "📋 КЛАССИФИКАЦИЯ (из базы классификаторов 1С СВКиА):\n\n"
    
    if selected and "deviation_category" in selected:
        dev_cat = selected["deviation_category"]
        classification_text += f"Категории отклонения:\n\n"
        # primary с обоснованием
        classification_text += f"1. {dev_cat.get('primary_id', 'N/A')}\n"
        classification_text += f"   Почему: {dev_cat.get('rationale', 'не указано')}\n\n"
        # первая альтернатива
        if "alternatives" in dev_cat and len(dev_cat["alternatives"]) >= 1:
            classification_text += f"2. {dev_cat['alternatives'][0]}\n\n"
    
    if selected and "risk" in selected:
        risk = selected["risk"]
        classification_text += f"Риски:\n\n"
        classification_text += f"1. {risk.get('primary_id', 'N/A')}\n"
        classification_text += f"   Почему: {risk.get('rationale', 'не указано')}\n\n"
        if "alternatives" in risk and len(risk["alternatives"]) >= 1:
            classification_text += f"2. {risk['alternatives'][0]}\n\n"
    
    await m.answer(classification_text)  # БЕЗ 
    
    txt = "\nВыберите варианты разделов:"
    await m.answer(txt, reply_markup=kb_sections(dev_id).as_markup())

# @dp.message(Command("export"))
# async def export_cmd(m: Message):
#     # ВРЕМЕННО ОТКЛЮЧЕНО
#     pass

@dp.callback_query(F.data.startswith("back|"))
async def cb_back(q: CallbackQuery):
    _, dev_id = q.data.split("|", 1)
    dev_id = int(dev_id)
    row = get_deviation(con, dev_id)
    selected = _loads(row.get("selected_json"))
    txt = (
        f"категория: {cat_title(selected.get('deviation_category',{}).get('primary_id',''))}\n"
        f"риск: {risk_title(selected.get('risk',{}).get('primary_id',''))}\n\n"
        "выбери раздел:"
    )
    await q.message.edit_text(txt, reply_markup=kb_sections(dev_id).as_markup(), )
    await q.answer()

@dp.callback_query(F.data.startswith("sec|"))
async def cb_section(q: CallbackQuery):
    _, dev_id, section_key = q.data.split("|", 2)
    dev_id = int(dev_id)
    row = get_deviation(con, dev_id)
    idx = get_chosen_variant(row, section_key)
    mode = get_view_mode(row, section_key)
    txt = render_section(row, section_key)
    await q.message.edit_text(txt, reply_markup=kb_section_controls(dev_id, section_key, idx, mode), )
    await q.answer()

@dp.callback_query(F.data.startswith("var|"))
async def cb_var(q: CallbackQuery):
    _, dev_id, section_key, idx = q.data.split("|", 3)
    dev_id = int(dev_id)
    idx = int(idx)
    row = get_deviation(con, dev_id)
    set_chosen_variant(con, dev_id, row, section_key, idx)
    row = get_deviation(con, dev_id)
    mode = get_view_mode(row, section_key)
    txt = render_section(row, section_key)
    await q.message.edit_text(txt, reply_markup=kb_section_controls(dev_id, section_key, idx, mode), )
    await q.answer("ок")

@dp.callback_query(F.data.startswith("mode|"))
async def cb_mode(q: CallbackQuery):
    _, dev_id, section_key = q.data.split("|", 2)
    dev_id = int(dev_id)
    row = get_deviation(con, dev_id)
    toggle_view_mode(con, dev_id, row, section_key)
    row = get_deviation(con, dev_id)
    idx = get_chosen_variant(row, section_key)
    mode = get_view_mode(row, section_key)
    txt = render_section(row, section_key)
    await q.message.edit_text(txt, reply_markup=kb_section_controls(dev_id, section_key, idx, mode), )
    await q.answer("переключил")

@dp.callback_query(F.data.startswith("custom|"))
async def cb_custom(q: CallbackQuery, state: FSMContext):
    _, dev_id, section_key = q.data.split("|", 2)
    await state.update_data(dev_id=int(dev_id), section_key=section_key)
    await state.set_state(CustomVariant.text_input)
    await q.message.answer("Введите свой текст для этого раздела:")
    await q.answer()

@dp.message(CustomVariant.text_input)
async def custom_text_input(m: Message, state: FSMContext):
    data = await state.get_data()
    dev_id = data["dev_id"]
    section_key = data["section_key"]
    custom_text = m.text
    
    # сохраняем custom вариант как 4-й (индекс 3)
    row = get_deviation(con, dev_id)
    sections_data = json.loads(row["sections_json"])
    
    if section_key in sections_data:
        # добавляем или заменяем 4-й вариант
        if len(sections_data[section_key]["variants"]) == 3:
            sections_data[section_key]["variants"].append(custom_text)
        else:
            sections_data[section_key]["variants"][3] = custom_text
        
        # обновляем в БД
        update_deviation(con, dev_id, sections=sections_data)
        
        # устанавливаем выбранный вариант на 4-й
        set_chosen_variant(con, dev_id, row, section_key, 3)
        
        row = get_deviation(con, dev_id)
        idx = 3
        mode = get_view_mode(row, section_key)
        txt = render_section(row, section_key)
        
        await m.answer(txt, reply_markup=kb_section_controls(dev_id, section_key, idx, mode), )
        await state.clear()
    else:
        await m.answer("Ошибка: раздел не найден")
        await state.clear()

@dp.callback_query(F.data.startswith("regen|"))
async def cb_regen(q: CallbackQuery):
    _, dev_id, section_key = q.data.split("|", 2)
    dev_id = int(dev_id)
    row = get_deviation(con, dev_id)
    if not row.get("sections_json"):
        await q.answer("сначала /build", show_alert=True)
        return

    ui = _loads(row.get("user_input_json"))
    selected = _loads(row.get("selected_json"))
    sections = _loads(row.get("sections_json"))

    # candidates for context (same as build)
    ui_obj = UserInput(**ui)
    candidates = topk_candidates(llm_embeddings, ui_obj, CATEGORIES, RISKS, CAT_MAT, RISK_MAT, k=20)

    await q.answer("regen…", show_alert=False)

    messages = [
        {"role": "system", "text": SYSTEM_PROMPT},
        {"role": "user", "text": build_regen_prompt(ui, selected, section_key, old_variants, candidates)},
    ]
    raw = llm.completion(messages, temperature=0.25, max_tokens=16000)

    # --- clean regen output ---
    clean_regen = raw.strip()
    if clean_regen.startswith("```"):
        clean_regen = clean_regen.split("\n", 1)[1] if "\n" in clean_regen else ""
        if "```" in clean_regen:
            clean_regen = clean_regen.rsplit("```", 1)[0]
    clean_regen = clean_regen.strip()
    # --------------------------

    try:
        data = json.loads(clean_regen)
        if data.get("section_key") != section_key:
            raise ValueError("section_key mismatch")
        new_variants = data["variants"]
        # basic shape validation + pad/cut to 5 variants
        if not isinstance(new_variants, list):
            raise ValueError("variants must be list")

        if len(new_variants) < 3:
            raise ValueError("variants must be list of 3..5")

        # pad to 5 if model returned only 3-4
        while len(new_variants) < 5:
            base = new_variants[-1] if new_variants else {"short": "", "full": ""}
            short = (base.get("short") or "").strip()
            full = (base.get("full") or "").strip()
            new_variants.append({
                "short": (short + " (вариант добавлен автоматически)") if short else "вариант добавлен автоматически",
                "full": (full + "\\n\\n(вариант добавлен автоматически: модель вернула меньше 5)") if full else "(вариант добавлен автоматически: модель вернула меньше 5)"
            })

        # cut to 5 if model returned more
        if len(new_variants) > 5:
            new_variants = new_variants[:5]

        sections[section_key] = {"variants": new_variants}
        update_deviation(con, dev_id, sections=sections)
    except Exception as e:
        await q.message.answer(f"regen упал: {e}\nсырец (до 900):\n{raw[:900]}")
        return

    row = get_deviation(con, dev_id)
    idx = get_chosen_variant(row, section_key)
    mode = get_view_mode(row, section_key)
    txt = render_section(row, section_key)
    await q.message.edit_text(txt, reply_markup=kb_section_controls(dev_id, section_key, idx, mode), )
    await q.answer("готово")

# @dp.callback_query(F.data.startswith("summary|"))
# async def cb_summary(q: CallbackQuery):
#     # ВРЕМЕННО ОТКЛЮЧЕНО
#     pass

@dp.callback_query(F.data.startswith("back|"))
async def cb_back(q: CallbackQuery):
    _, dev_id = q.data.split("|", 1)
    dev_id = int(dev_id)
    txt = "Выберите варианты разделов. Выбранный вариант (✓) попадёт в итоговое резюме:"
    await q.message.answer(txt, reply_markup=kb_sections(int(dev_id)).as_markup())
    await q.answer()

# @dp.callback_query(F.data.startswith("export|"))
# async def cb_export(q: CallbackQuery):
#     # ВРЕМЕННО ОТКЛЮЧЕНО
#     pass

async def main():
    await dp.start_polling(bot)


@dp.message(F.text)
async def check_code(m: Message):
    # проверка кода доступа
    if not is_authorized(m.from_user.id):
        if m.text.strip() == ACCESS_CODE:
            authorize(m.from_user.id)
            await m.answer(
                "✅ Доступ разрешён на сегодня!\n\n"
                "⚠️ ВНИМАНИЕ - ОТКАЗ ОТ ОТВЕТСТВЕННОСТИ:\n\n"
                "Данный бот НЕ предназначен для обработки информации, содержащей:\n"
                "• Государственную тайну (ФЗ-5487-1)\n"
                "• Коммерческую тайну (ФЗ-98)\n"
                "• Персональные данные (ФЗ-152)\n"
                "• Инсайдерскую информацию (ФЗ-224)\n\n"
                "Команды: /new, /build N, /preview N",
                reply_markup=main_menu()
            )
        else:
            await m.answer("❌ Неверный код")
        return
    
    # если это не команда - считаем что это описание отклонения
    if not m.text.startswith("/"):
        return  # пропускаем, обработается в другом месте


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
# === АВТОРИЗАЦИЯ ===
ACCESS_CODE = "audit_2026"
authorized_users = {}  # {user_id: дата авторизации}

from datetime import date

def is_authorized(user_id):
    if user_id not in authorized_users:
        return False
    return authorized_users[user_id] == date.today()

def authorize(user_id):
    authorized_users[user_id] = date.today()




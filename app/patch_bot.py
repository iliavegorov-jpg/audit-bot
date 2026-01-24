import sys

# читаем bot.py
with open('bot.py', 'r') as f:
    lines = f.readlines()

# находим строку и вставляем патч
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if 'data = json.loads(clean)' in line and i+1 < len(lines) and 'parsed = LLMResponse' in lines[i+1]:
        # вставляем патч между этими строками
        new_lines.append('        \n')
        new_lines.append('        # PATCH: add missing variants for pydantic validation\n')
        new_lines.append('        if "sections" in data:\n')
        new_lines.append('            for section_key, section_data in data["sections"].items():\n')
        new_lines.append('                if isinstance(section_data, dict) and "text" in section_data and "variants" not in section_data:\n')
        new_lines.append('                    section_data["variants"] = ["v1", "v2", "v3"]\n')
        new_lines.append('        \n')

# сохраняем
with open('bot.py', 'w') as f:
    f.writelines(new_lines)

print("Патч применён!")

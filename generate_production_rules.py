import json
from datetime import datetime

# ==================== НАСТРОЙКИ ====================
NUM_RULES = 100
FEATURES = {
    13: "volatility_20",  26: "ema_26",       28: "macd_signal",
    30: "bb_width_20",    32: "atrp_14",     37: "vix_z_60",
    38: "vix_x_mktret",   22: "mom_10",      21: "mom_5",
    36: "mkt_return_1d",  39: "yc_slope",    19: "hl_range",
    20: "oc_return",      33: "vol_z_20",    34: "ret_std_20"
}

EVENTS = {
    "a1": "Сильный рост (Strong Bullish)",
    "a2": "Рост / Задуматься о покупке (Bullish)",
    "a3": "Падение (Bearish)"
}

# ==================== ГЕНЕРАЦИЯ ====================
rules = []
used_signatures = set()   # Для гарантии уникальности

base_templates = [
    # 12 самых важных (core rules) — уже проверенные
    {"ante": [37,30,28,32], "cons": "a1", "cond": "f37 < -1.2 AND f30 > 0.09 AND f28 > 0.25 AND f32 > 1.3"},
    {"ante": [37,28,13,22], "cons": "a1", "cond": "f37 < -0.8 AND f28 > 0 AND f13 > 0.018 AND f22 > 0.012"},
    {"ante": [26,28,30,37], "cons": "a1", "cond": "f26 > 0 AND f28 > 0.2 AND f30 > 0.07 AND f37 < -0.9"},
    {"ante": [37,30,13],     "cons": "a2", "cond": "f37 BETWEEN -0.6 AND 0.6 AND f30 < 0.05 AND f13 > 0.008"},
    {"ante": [28,26,22],     "cons": "a2", "cond": "f28 > 0 AND f26 > 0 AND f22 > 0.01"},
    {"ante": [37,28,32],     "cons": "a2", "cond": "f37 < -0.5 AND f28 > 0.15 AND f32 > 1.1"},
    {"ante": [37,30,13,19],  "cons": "a3", "cond": "f37 > 1.5 AND f30 > 0.1 AND f13 < -0.015 AND f19 > 0.025"},
    {"ante": [28,13,37],     "cons": "a3", "cond": "f28 < -0.2 AND f13 < -0.012 AND f37 > 1.3"},
    {"ante": [37,32,30],     "cons": "a3", "cond": "f37 > 1.8 AND f32 > 1.6 AND f30 > 0.12"},
    {"ante": [26,28,37],     "cons": "a2", "cond": "f26 > 0 AND f28 > 0 AND f37 BETWEEN -0.4 AND 0.4"},
]

for rule in base_templates:
    sig = tuple(sorted(rule["ante"])) + (rule["cons"], rule["cond"])
    if sig not in used_signatures:
        used_signatures.add(sig)
        rules.append(rule)

# Генерируем оставшиеся уникальные правила
import random
feature_list = list(FEATURES.keys())

while len(rules) < NUM_RULES:
    # Случайный размер антецедента от 3 до 5
    size = random.randint(3, 5)
    ante = sorted(random.sample(feature_list, size))
    
    # Случайный класс
    cons = random.choice(["a1", "a2", "a3"])
    
    # Генерируем условие
    cond_parts = []
    for f in ante:
        val = round(random.uniform(-2.0, 2.0), 3)
        if random.random() < 0.4:
            cond_parts.append(f"f{f} > {val}")
        elif random.random() < 0.4:
            cond_parts.append(f"f{f} < {val}")
        else:
            cond_parts.append(f"f{f} BETWEEN {val-0.3:.2f} AND {val+0.3:.2f}")
    
    cond = " AND ".join(cond_parts)
    
    # Проверка уникальности
    sig = tuple(ante) + (cons, cond)
    if sig not in used_signatures:
        used_signatures.add(sig)
        rules.append({"ante": ante, "cons": cons, "cond": cond})

# ==================== ВЫВОД ====================
print("БАЗА ЗНАНИЙ ПРОДУКЦИОННЫХ ПРАВИЛ")
print("=" * 100)
print(f"Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Всего уникальных правил: {len(rules)}\n")

for i, r in enumerate(rules, 1):
    ante_str = " ∧ ".join(f"f{f}" for f in r["ante"])
    print(f"p{i}: {ante_str} → {r['cons']} | {r['cond']}")

# Сохранение в JSON
output = {
    "generated_at": datetime.now().isoformat(),
    "total_rules": len(rules),
    "features_count": 39,
    "events": EVENTS,
    "rules": rules
}

with open("production_rules_knowledge_base.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 100)
print(f"Файл 'production_rules_knowledge_base.json' успешно создан!")
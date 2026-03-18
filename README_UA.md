# Anima — Ollama Integration

![Банер Subjective Core](v4.jpg)

> Емоційний AI-агент з суб'єктивним внутрішнім станом, що працює повністю локально через [Ollama](https://ollama.com).

---

## Що це таке

**Anima** — це Python-реалізація AI-агента, у якого є змодельована емоційна система.  
Агент не просто генерує текст — він переживає стимули, формує спогади, будує внутрішній нарратив і лише потім відповідає. Кожна відповідь LLM забарвлена поточним емоційним станом агента.

Проєкт побудований на кількох теоретичних моделях:

| Концепція | Реалізація |
|---|---|
| VAD (Valence-Arousal-Dominance) | Тривимірний простір емоцій |
| Big Five / OCEAN | Модель особистості агента |
| IIT (Integrated Information Theory) | Наближення φ-інтеграції |
| Predictive Processing (Friston) | Похибка передбачення, вільна енергія |
| Plutchik's Wheel | Вісім базових емоцій |
| Confabulation | Хибне спогадування |

---

## Архітектура

```
AnimaOllama                   ← головний клас
│
├── OllamaBridge              ← REST-клієнт до Ollama
├── Personality               ← риси Big Five + confabulation rate
│
├── AdaptiveEmotionMap        ← карта VAD → емоція (навчається)
├── AssociativeMemory         ← асоціативна пам'ять (cosine similarity)
├── IITModule                 ← обчислення φ (Tononi)
├── PredictiveProcessor       ← prediction error + free energy
├── HomeostaticDrive          ← гомеостатичні потяги
└── ExistentialNarrative      ← нарратив від першої особи
```

### Цикл обробки стимулу

```
stimulus (Dict)
    ↓
[Confabulation?]  ← випадкові хибні спогади
    ↓
[Memory Resonance] ← спогади підсилюють стимул
    ↓
[Apply + Decay]   ← реактори: tension, arousal, satisfaction, cohesion
    ↓
[VAD Vector]      ← valence = satisfaction - tension, etc.
    ↓
[Emotion ID]      ← найближча емоція у VAD-просторі
    ↓
[Filter]          ← трансформація (Смуток+напруга → Гнів-захист)
    ↓
[IIT φ]           ← ентропія цілого vs частин
    ↓
[Prediction Error] ← здивування, вільна енергія
    ↓
[Drives]          ← гомеостатичні потяги
    ↓
[Narrative]       ← внутрішній нарратив від першої особи
    ↓
state Dict → OllamaBridge.build_system_prompt() → LLM → відповідь
```

---

## Встановлення та запуск

### 1. Встановити Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows — завантажити інсталятор з https://ollama.com
```

### 2. Завантажити модель

```bash
ollama pull llama3        # рекомендовано (~4.7 GB)
# або
ollama pull mistral       # (~4.1 GB)
ollama pull gemma2        # (~5.4 GB)
ollama pull phi3          # (~2.3 GB, швидка)
ollama pull qwen2.5       # (~4.4 GB)
```

### 3. Встановити Python-залежності

```bash
pip install requests numpy
```

### 4. Запустити

```bash
python Anima_ollama.py              # використовує llama3
python Anima_ollama.py mistral      # вказати іншу модель
python Anima_ollama.py phi3         # легша модель
```

---

## Інтерактивні команди

| Команда | Дія |
|---|---|
| `(текст)` | Звичайне повідомлення — агент відповідає з урахуванням стану |
| `/stress` | Підвищити напругу та збудженість |
| `/relax` | Знизити напругу, розслабитись |
| `/connect` | Відчути підтримку та зв'язок |
| `/shock` | Раптовий сильний стрес |
| `/state` | Показати поточний емоційний стан |
| `/export` | Зберегти журнал станів → `state_log.json` |
| `/reset` | Скинути реактори до базових значень |
| `/models` | Список завантажених Ollama-моделей |
| `/quit` | Вийти |

---

## Використання як бібліотека

```python
from anima_ollama import AnimaOllama, Personality

# Налаштувати особистість
persona = Personality(
    neuroticism=0.7,       # схильний до тривоги
    extraversion=0.4,      # інтроверт
    agreeableness=0.8,     # дуже доброзичливий
    conscientiousness=0.6,
    openness=0.9,          # дуже відкритий до нового
    confabulation_rate=0.5,
)

agent = AnimaOllama(
    personality=persona,
    ollama_model="mistral",
    temperature=0.8,
)

# Застосувати стимул вручну
state = agent.experience({"tension": 0.3, "arousal": 0.2})
print(state["primary"])      # → "Очікування"
print(state["narrative"])    # → "Щось має статись..."
print(state["phi"])          # → 0.42

# Отримати відповідь від LLM
response = agent.chat(
    "Як справи?",
    stimulus={"cohesion": 0.1},
)
print(response)

# Зберегти журнал
agent.export_history("my_session.json")
```

---

## Структура стану (`experience()` → Dict)

```python
{
    "primary":          "Очікування",          # первинна емоція
    "blend": [
        {"name": "Очікування", "intensity": 1.0},
        {"name": "Довіра",     "intensity": 0.3},
    ],
    "vad": {
        "valence":   0.25,   # -1 (негативне) ↔ +1 (позитивне)
        "arousal":   0.38,   # -1 (спокій) ↔ +1 (збудження)
        "dominance": 0.42,   # -1 (безсилля) ↔ +1 (контроль)
    },
    "reactors": {
        "tension":      0.22,
        "arousal":      0.31,
        "satisfaction": 0.48,
        "cohesion":     0.55,
    },
    "phi":              0.38,   # IIT φ-інтеграція
    "phi_label":        "низька інтеграція",
    "prediction_error": 0.14,   # похибка передбачення
    "pred_label":       "незначне відхилення",
    "free_energy":      0.09,
    "surprise_spike":   False,
    "dominant_drive":   None,   # або "cohesion" / "tension" / etc.
    "narrative":        "Щось має статись. Увага спрямована вперед.",
    "memory_resonance": 2,      # кількість активованих спогадів
}
```

---

## Реактори та їх значення

Чотири реактори — це «внутрішні виміри» стану агента:

| Реактор | Базове | Опис |
|---|---|---|
| `tension` | 0.2 | Внутрішня напруга, стрес |
| `arousal` | 0.2 | Збудженість, активність |
| `satisfaction` | 0.5 | Задоволеність, спокій |
| `cohesion` | 0.5 | Відчуття зв'язку, приналежності |

З реакторів будується вектор VAD:
```
valence   = satisfaction - tension
arousal   = arousal + tension * 0.3
dominance = cohesion + (satisfaction - tension) * 0.5
```

---

## Особистість (Big Five)

| Параметр | Вплив |
|---|---|
| `neuroticism` | Підсилює реакцію на стрес; Смуток → Гнів-захист |
| `extraversion` | Підсилює збудженість |
| `agreeableness` | Підсилює cohesion; Гнів → Асертивність |
| `conscientiousness` | Прискорює decay (повернення до базового) |
| `openness` | Підвищує чутливість до здивування |
| `confabulation_rate` | Ймовірність хибних спогадів |

---

## Вимоги

- Python 3.9+
- `numpy`
- `requests`
- Ollama (запущений локально)

---

## Ліцензія

Цей проєкт розповсюджується на умовах ліцензії **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

- **Особисте та наукове використання:** Ви можете вільно використовувати, змінювати та розвивати цей код для некомерційних досліджень або власних проєктів.
- **Комерційне та корпоративне використання:** **Суворо заборонено.** Будь-яке використання прибутковими організаціями або інтеграція в комерційні продукти/сервіси потребує окремої комерційної ліцензії та письмового дозволу автора.
- **Атрибуція:** Ви обов'язково повинні вказувати авторство оригінального розробника.

Copyright (c) 2026 [Stell]

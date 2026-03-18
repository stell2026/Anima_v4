"""
╔══════════════════════════════════════════════════════════════════╗
║                Anima v4  —  OLLAMA ІНТЕГРАЦІЯ                    ║
║                                                                  ║
║  Як запустити:                                                   ║
║    1. Встановити Ollama: https://ollama.com                      ║
║    2. Завантажити модель: ollama pull llama3  (або mistral,      ║
║                           gemma2, phi3, qwen2.5 тощо)            ║
║    3. pip install requests numpy                                 ║
║    4. python anima_ollama.py                                     ║
║    5. (опційно) anima_ollama.py mistral                          ║
╚══════════════════════════════════════════════════════════════════╝

Архітектура:
  OllamaBridge          — комунікація з локальною LLM через REST API
  Personality           — модель особистості (Big Five / OCEAN)
  MemoryTrace           — одиниця асоціативної пам'яті
  AssociativeMemory     — сховище та пошук спогадів
  AdaptiveEmotionMap    — адаптивна карта емоцій у просторі VAD
  IITModule             — наближення φ (Integrated Information Theory)
  PredictiveProcessor   — предиктивна обробка та вільна енергія
  HomeostaticDrive      — гомеостатичні потяги та потреби
  ExistentialNarrative  — генератор внутрішнього нарративу
  AnimaOllama           — головний клас агента
"""

import numpy as np
import time
import json
import requests
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque


# ══════════════════════════════════════════════════════════════════
# ЛОГУВАННЯ
# ══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# OLLAMA BRIDGE
# ══════════════════════════════════════════════════════════════════

class OllamaBridge:
    """
    Комунікація з локальною Ollama через REST API.
    Ollama слухає на http://localhost:11434 за замовчуванням.

    Сумісні моделі (рекомендовані для емоційного агента):
      llama3, llama3.1, mistral, gemma2, phi3, qwen2.5, deepseek-r1

    Атрибути:
        model (str):       назва моделі Ollama
        base_url (str):    базова URL Ollama сервера
        temperature (float): температура генерації (0.0–1.0)
        timeout (int):     максимальний час очікування відповіді (секунди)
    """

    DEFAULT_URL   = "http://localhost:11434"
    DEFAULT_MODEL = "llama3"

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_URL,
                 temperature: float = 0.75,
                 timeout: int = 60):
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout     = timeout

    def is_available(self) -> bool:
        """Повертає True якщо Ollama сервер доступний."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Повертає список завантажених моделей Ollama."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def chat(self, system_prompt: str, user_message: str) -> str:
        """
        Надсилає запит до Ollama через /api/chat.

        Args:
            system_prompt (str): системний промпт з емоційним контекстом
            user_message (str):  повідомлення користувача

        Returns:
            str: відповідь моделі або повідомлення про помилку
        """
        payload = {
            "model":  self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
        }
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.exceptions.ConnectionError:
            return ("[OLLAMA НЕДОСТУПНА]\n"
                    "Запустіть: ollama serve\n"
                    "Завантажте модель: ollama pull llama3")
        except requests.exceptions.Timeout:
            return f"[OLLAMA ТАЙМАУТ] Відповідь не отримана за {self.timeout}с."
        except requests.exceptions.HTTPError as e:
            return f"[OLLAMA HTTP ПОМИЛКА] {e}"
        except (KeyError, json.JSONDecodeError) as e:
            return f"[OLLAMA НЕОЧІКУВАНА ВІДПОВІДЬ] {e}"
        except Exception as e:
            logger.exception("Невідома помилка Ollama")
            return f"[OLLAMA ПОМИЛКА] {e}"

    def build_system_prompt(self,
                            state: Dict,
                            personality: "Personality",
                            memories: List) -> str:
        """
        Будує системний промпт, що кодує поточний емоційний стан агента.

        Args:
            state (Dict):          результат AnimaOllama.experience()
            personality (Personality): риси особистості агента
            memories (List[MemoryTrace]): нещодавні спогади

        Returns:
            str: готовий системний промпт для LLM
        """
        blend = " + ".join(
            f"{e['name']} ({e['intensity']:.0%})"
            for e in state.get("blend", []))

        mem_s = ""
        if memories:
            mem_s = "Recent memory echoes: " + "; ".join(
                f"{m.emotion} (weight {m.weight:.1f})" for m in memories)

        # Промпт англійською — більшість локальних моделей краще
        # розуміють емоційні інструкції англійською
        return f"""You are an emotionally aware AI agent with a rich inner life.
Your current subjective state (do NOT mention these numbers directly,
but let them shape your tone, word choice, and response length):

<inner_state>
  primary_emotion  : {state.get('primary', 'neutral')}
  emotional_blend  : {blend}
  valence          : {state.get('vad', {}).get('valence', 0):+.2f}   (negative ↔ positive)
  arousal          : {state.get('vad', {}).get('arousal', 0):+.2f}   (calm ↔ excited)
  dominance        : {state.get('vad', {}).get('dominance', 0):+.2f} (powerless ↔ in control)
  phi_IIT          : {state.get('phi', 0):.2f}   (integration of consciousness)
  prediction_error : {state.get('prediction_error', 0):.2f}  (surprise level)
  free_energy      : {state.get('free_energy', 0):.2f}
  dominant_drive   : {state.get('dominant_drive') or 'none'}
  inner_narrative  : "{state.get('narrative', '')}"
  personality:
    neuroticism={personality.neuroticism:.1f}
    extraversion={personality.extraversion:.1f}
    agreeableness={personality.agreeableness:.1f}
  {mem_s}
</inner_state>

Guidelines:
- High arousal (>0.6) → shorter, more energetic sentences
- Negative valence (<-0.3) → cautious, careful, or withdrawn tone
- High prediction_error (>0.5) → express confusion or surprise naturally
- dominant_drive = cohesion → express need for connection
- dominant_drive = tension → express tiredness or overwhelm
- NEVER quote numbers or tag names. Express state through language only.
- Respond in the same language the user writes in."""

    def respond(self,
                user_message: str,
                state: Dict,
                personality: "Personality",
                memories: List) -> str:
        """
        Генерує відповідь з урахуванням емоційного стану.

        Args:
            user_message (str): повідомлення від користувача
            state (Dict):       поточний стан агента
            personality:        риси особистості
            memories:           активовані спогади

        Returns:
            str: відповідь LLM
        """
        system = self.build_system_prompt(state, personality, memories)
        return self.chat(system, user_message)


# ══════════════════════════════════════════════════════════════════
# МОДЕЛІ ДАНИХ
# ══════════════════════════════════════════════════════════════════

@dataclass
class Personality:
    """
    Модель особистості на основі Big Five (OCEAN).

    Атрибути:
        neuroticism (float):        нейротизм — схильність до негативних емоцій [0–1]
        extraversion (float):       екстраверсія — енергійність, товариськість [0–1]
        agreeableness (float):      доброзичливість — схильність до кооперації [0–1]
        conscientiousness (float):  сумлінність — організованість, самодисципліна [0–1]
        openness (float):           відкритість — цікавість, творчість [0–1]
        confabulation_rate (float): ймовірність генерації хибних спогадів [0–1]
    """
    neuroticism:        float = 0.5
    extraversion:       float = 0.5
    agreeableness:      float = 0.5
    conscientiousness:  float = 0.5
    openness:           float = 0.5
    confabulation_rate: float = 0.8

    def tension_multiplier(self)   -> float:
        """Нейротизм посилює тривогу та стрес."""
        return 1.0 + (self.neuroticism   - 0.5) * 0.8

    def arousal_multiplier(self)   -> float:
        """Екстраверсія посилює збудженість."""
        return 1.0 + (self.extraversion  - 0.5) * 0.6

    def cohesion_multiplier(self)  -> float:
        """Доброзичливість підсилює соціальну прив'язаність."""
        return 1.0 + (self.agreeableness - 0.5) * 0.6

    def decay_rate(self)           -> float:
        """Сумлінність прискорює повернення до базового стану."""
        return 0.1  + self.conscientiousness * 0.15

    def surprise_sensitivity(self) -> float:
        """Відкритість підвищує чутливість до несподіваного."""
        return 0.5  + self.openness * 0.5


@dataclass
class MemoryTrace:
    """
    Один слід асоціативної пам'яті.

    Атрибути:
        stimulus (Dict):    вектор стимулу, що викликав спогад
        emotion (str):      первинна емоція в момент запам'ятовування
        vad (np.ndarray):   вектор VAD (valence, arousal, dominance)
        intensity (float):  інтенсивність емоції [0–1]
        timestamp (str):    час створення спогаду
        weight (float):     вага (підсилюється при повторному зустрічі)
    """
    stimulus:  Dict
    emotion:   str
    vad:       np.ndarray
    intensity: float
    timestamp: str
    weight:    float = 1.0

    def similarity(self, other: Dict) -> float:
        """
        Косинусна подібність між збереженим і поточним стимулами.

        Args:
            other (Dict): поточний стимул

        Returns:
            float: подібність від 0.0 до 1.0
        """
        keys = set(self.stimulus) & set(other)
        if not keys:
            return 0.0
        a = np.array([self.stimulus.get(k, 0) for k in keys])
        b = np.array([other.get(k, 0)         for k in keys])
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / norm) if norm > 0 else 0.0


# ══════════════════════════════════════════════════════════════════
# ПІДСИСТЕМИ
# ══════════════════════════════════════════════════════════════════

class AssociativeMemory:
    """
    Асоціативна пам'ять з обмеженою ємністю (MAX_TRACES записів).

    Нові записи, що схожі на існуючі (> 0.85), підсилюють їх вагу
    замість додавання дубліката. Спогади з низькою подібністю (<0.3)
    ігноруються при пошуку.
    """
    MAX_TRACES = 200

    def __init__(self):
        self.traces: deque[MemoryTrace] = deque(maxlen=self.MAX_TRACES)

    def store(self,
              stimulus: Dict,
              emotion: str,
              vad: np.ndarray,
              intensity: float) -> None:
        """
        Зберігає новий слід або підсилює схожий існуючий.

        Args:
            stimulus:  поточний стимул
            emotion:   назва емоції
            vad:       вектор VAD
            intensity: інтенсивність емоції
        """
        for t in self.traces:
            if t.similarity(stimulus) > 0.85:
                t.weight = min(t.weight + 0.2, 3.0)
                return
        self.traces.append(MemoryTrace(
            dict(stimulus), emotion,
            vad.copy(), intensity,
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ))

    def recall(self, stimulus: Dict, top_k: int = 3) -> List[MemoryTrace]:
        """
        Повертає top_k найбільш релевантних спогадів.

        Args:
            stimulus: поточний стимул для пошуку
            top_k:    кількість спогадів для повернення

        Returns:
            List[MemoryTrace]: відсортовані за релевантністю сліди
        """
        scored = [(t, t.similarity(stimulus) * t.weight) for t in self.traces]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, s in scored[:top_k] if s > 0.3]

    def resonance_delta(self, stimulus: Dict) -> Dict:
        """
        Обчислює вплив спогадів на поточний стан реакторів.

        Args:
            stimulus: поточний стимул

        Returns:
            Dict: дельта для кожного реактора
        """
        delta: Dict[str, float] = {}
        for mem in self.recall(stimulus):
            for key, val in mem.stimulus.items():
                delta[key] = delta.get(key, 0.0) + val * 0.1 * mem.weight
        return delta

    def __len__(self) -> int:
        return len(self.traces)


class AdaptiveEmotionMap:
    """
    Адаптивна карта емоцій у просторі VAD (Valence-Arousal-Dominance).

    Вектори восьми базових емоцій Плутчика повільно навчаються через
    досвід (LEARN_RATE) і повертаються до базових значень (DECAY_RATE).
    """
    BASE_MAP = {
        "Страх":      np.array([-0.8,  0.8, -0.7]),
        "Гнів":       np.array([-0.6,  0.9,  0.8]),
        "Радість":    np.array([ 0.9,  0.5,  0.6]),
        "Смуток":     np.array([-0.7, -0.5, -0.6]),
        "Здивування": np.array([ 0.2,  1.0,  0.1]),
        "Огида":      np.array([-0.7,  0.2,  0.4]),
        "Очікування": np.array([ 0.3,  0.4,  0.5]),
        "Довіра":     np.array([ 0.7, -0.1,  0.3]),
    }
    LEARN_RATE = 0.03
    DECAY_RATE = 0.005

    def __init__(self):
        self.vectors   = {k: v.copy() for k, v in self.BASE_MAP.items()}
        self.frequency = {k: 0        for k in self.BASE_MAP}

    def identify(self, vad: np.ndarray, top_k: int = 2) -> List[Dict]:
        """
        Ідентифікує найближчі емоції до заданого VAD вектора.

        Args:
            vad:   вектор VAD
            top_k: кількість емоцій для повернення

        Returns:
            List[Dict]: [{name, distance, intensity}, ...]
        """
        distances = [
            {"name": n, "distance": float(np.linalg.norm(vad - v))}
            for n, v in self.vectors.items()
        ]
        distances.sort(key=lambda x: x["distance"])
        top = distances[:top_k]
        max_d = top[-1]["distance"] or 1.0
        for item in top:
            item["intensity"] = round(1.0 - item["distance"] / (max_d + 1e-9), 3)
        return top

    def learn(self, name: str, vad: np.ndarray) -> None:
        """Оновлює вектор емоції через градієнтний крок."""
        if name not in self.vectors:
            return
        self.frequency[name] += 1
        self.vectors[name] += self.LEARN_RATE * (vad - self.vectors[name])
        self.vectors[name]   = np.clip(self.vectors[name], -1.0, 1.0)

    def decay_toward_base(self) -> None:
        """Повільно повертає всі вектори до базових значень."""
        for name in self.vectors:
            self.vectors[name] += self.DECAY_RATE * (self.BASE_MAP[name] - self.vectors[name])

    def get_state(self) -> Dict:
        """Повертає поточний стан векторів (для серіалізації)."""
        return {k: v.tolist() for k, v in self.vectors.items()}


class IITModule:
    """
    Наближення інтегрованої інформації (φ) за Тонові (IIT).

    φ > 0 означає, що ціле є більшим за суму частин —
    тобто система демонструє певну форму інтеграції стану.
    Це груба апроксимація, не повна теорія IIT.
    """

    @staticmethod
    def compute(vad: np.ndarray, reactors: Dict) -> float:
        """
        Обчислює φ як різницю між ентропією цілого та частин.

        Args:
            vad:      вектор VAD
            reactors: словник реакторів

        Returns:
            float: значення φ ∈ [0, ∞)
        """
        def entropy(arr: np.ndarray) -> float:
            p = np.abs(arr)
            s = p.sum()
            if s < 1e-9:
                return 0.0
            p = p / s
            p = p[p > 0]
            return float(-np.sum(p * np.log2(p + 1e-12)))

        h_whole = entropy(vad)
        h_parts = sum(entropy(np.array([v])) for v in reactors.values())
        return round(max(0.0, h_whole - h_parts * 0.25), 4)

    @staticmethod
    def interpret(phi: float) -> str:
        """Повертає текстову інтерпретацію значення φ."""
        if phi < 0.05: return "мінімальна інтеграція"
        if phi < 0.20: return "низька інтеграція"
        if phi < 0.50: return "середня інтеграція"
        if phi < 0.80: return "висока інтеграція"
        return               "максимальна інтеграція (потік)"


class PredictiveProcessor:
    """
    Предиктивна обробка та вільна енергія (за Карлом Фрістоном).

    Агент будує внутрішню модель і обчислює похибку передбачення
    між очікуваним та фактичним VAD.
    Вільна енергія — це середня похибка за останні 50 кроків.
    """

    def __init__(self):
        self.predicted_vad: Optional[np.ndarray] = None
        self.error_history: deque[float]         = deque(maxlen=50)

    def predict(self, vad: np.ndarray) -> None:
        """Зберігає поточний VAD як передбачення наступного стану."""
        self.predicted_vad = vad.copy()

    def compute_error(self,
                      actual_vad: np.ndarray,
                      sensitivity: float = 1.0) -> Tuple[float, str]:
        """
        Обчислює похибку між передбаченим та фактичним VAD.

        Args:
            actual_vad:  фактичний вектор VAD
            sensitivity: коефіцієнт чутливості (з openness особистості)

        Returns:
            Tuple[float, str]: (похибка, текстова мітка)
        """
        if self.predicted_vad is None:
            self.error_history.append(0.0)
            return 0.0, "немає передбачення"
        error = round(
            float(np.linalg.norm(actual_vad - self.predicted_vad)) * sensitivity, 4
        )
        self.error_history.append(error)
        if error < 0.05:   label = "підтвердження"
        elif error < 0.20: label = "незначне відхилення"
        elif error < 0.50: label = "помітне здивування"
        elif error < 0.80: label = "порушення моделі"
        else:              label = "шок"
        return error, label

    def free_energy(self) -> float:
        """Повертає середню похибку передбачення (вільна енергія)."""
        return round(float(np.mean(self.error_history)), 4) if self.error_history else 0.0

    def surprise_spike(self) -> bool:
        """True якщо поточна похибка вдвічі вища за середню."""
        if len(self.error_history) < 3:
            return False
        return self.error_history[-1] > np.mean(list(self.error_history)[:-1]) * 2.0


class HomeostaticDrive:
    """
    Гомеостатичні потяги — відхилення реакторів від базових значень.

    Якщо реактор відхиляється більше ніж на THRESHOLD від базового
    рівня, виникає потяг (drive) — суб'єктивна потреба.
    """
    BASELINE  = {"tension": 0.2, "arousal": 0.2, "satisfaction": 0.5, "cohesion": 0.5}
    THRESHOLD = 0.4
    NEEDS     = {
        "tension":      "відпочинок",
        "arousal":      "заспокоїтись",
        "satisfaction": "ресурс",
        "cohesion":     "зв'язок",
    }

    def compute(self, reactors: Dict) -> Dict:
        """
        Обчислює активні потяги.

        Returns:
            Dict: {reactor_name: {intensity, direction, need}}
        """
        return {
            k: {
                "intensity": round(abs(reactors[k] - b), 3),
                "direction": "надлишок" if reactors[k] > b else "нестача",
                "need":      self.NEEDS[k],
            }
            for k, b in self.BASELINE.items()
            if abs(reactors[k] - b) > self.THRESHOLD
        }

    def dominant(self, reactors: Dict) -> Optional[str]:
        """Повертає назву найсильнішого поточного потягу."""
        drives = self.compute(reactors)
        return max(drives, key=lambda k: drives[k]["intensity"]) if drives else None


class ExistentialNarrative:
    """
    Генератор внутрішнього нарративу агента від першої особи.

    Обирає базову фразу відповідно до первинної емоції і доповнює її
    залежно від рівня φ, похибки передбачення та домінуючого потягу.
    """
    TEMPLATES: Dict[str, List[str]] = {
        "Страх":                   ["Щось загрожує. Треба або боротись, або тікати."],
        "Гнів":                    ["Мене порушили. Кордони були перетнуті."],
        "Гнів (захисна реакція)":  ["Мені боляче, але я показую силу."],
        "Радість":                 ["Все складається. Є відчуття повноти."],
        "Смуток":                  ["Чогось бракує. Або когось."],
        "Здивування":              ["Цього я не очікував. Модель світу перебудовується."],
        "Огида":                   ["Це не моє. Хочу дистанції."],
        "Очікування":              ["Щось має статись. Увага спрямована вперед."],
        "Довіра":                  ["Тут безпечно. Є відчуття опори."],
        "Тривога":                 ["Нічого конкретного, але тривожно."],
        "Оціпеніння":              ["Занадто багато. Система відключилась від болю."],
        "Асертивність":            ["Я знаю чого хочу. Скажу це спокійно."],
        "Полегшення":              ["Добре, але ще не до кінця вірю."],
        "Рішучість":               ["Ситуація вимагає дії. Я готовий."],
    }
    DEFAULT = ["Стан невизначений. Спостерігаю за собою."]

    def generate(self,
                 primary: str,
                 phi: float,
                 pred_error: float,
                 drive: Optional[str]) -> str:
        """
        Генерує текстовий нарратив для поточного стану.

        Args:
            primary:    первинна емоція
            phi:        рівень φ (інтеграція)
            pred_error: похибка передбачення
            drive:      домінуючий потяг або None

        Returns:
            str: нарратив від першої особи
        """
        options = self.TEMPLATES.get(primary, self.DEFAULT)
        base    = options[int(time.time()) % len(options)]
        notes   = []
        if phi > 0.5:
            notes.append("Усвідомлення чітке.")
        if pred_error > 0.5:
            notes.append("Щось пішло не так, як я думав.")
        if drive == "cohesion":
            notes.append("Хочу бути з кимось.")
        elif drive == "tension":
            notes.append("Потрібен відпочинок.")
        elif drive == "satisfaction":
            notes.append("Чогось не вистачає.")
        return base + (" " + " ".join(notes) if notes else "")


# ══════════════════════════════════════════════════════════════════
# ГОЛОВНИЙ КЛАС
# ══════════════════════════════════════════════════════════════════

class AnimaOllama:
    """
    Емоційний агент із суб'єктивним ядром та Ollama LLM.

    Об'єднує всі підсистеми: емоційну карту, пам'ять, IIT, предиктивну
    обробку, гомеостатичні потяги та нарративний генератор.

    Основний цикл:
        1. experience(stimulus) → обробляє стимул, повертає стан
        2. chat(message, stimulus) → генерує LLM-відповідь зі стану

    Атрибути:
        reactors (Dict):        поточні значення чотирьох реакторів
        personality (Personality): риси особистості
        memory (AssociativeMemory): асоціативна пам'ять
        identity_stream (List): хронологічний журнал станів
        llm (OllamaBridge):     міст до Ollama
    """

    BASELINE = {
        "tension":      0.2,
        "arousal":      0.2,
        "satisfaction": 0.5,
        "cohesion":     0.5,
    }
    MAX_STREAM = 100

    def __init__(self,
                 personality: Optional[Personality] = None,
                 ollama_model: str = "llama3",
                 ollama_url:   str = "http://localhost:11434",
                 temperature:  float = 0.75):
        """
        Args:
            personality (Personality): риси особистості; якщо None — дефолтна
            ollama_model (str):        назва моделі Ollama
            ollama_url (str):          URL Ollama сервера
            temperature (float):       температура генерації LLM
        """
        self.reactors      = dict(self.BASELINE)
        self.emotion_map   = AdaptiveEmotionMap()
        self.personality   = personality or Personality()
        self.memory        = AssociativeMemory()
        self.iit           = IITModule()
        self.predictor     = PredictiveProcessor()
        self.drive         = HomeostaticDrive()
        self.narrative_gen = ExistentialNarrative()
        self.identity_stream: List[Dict] = []

        self.llm = OllamaBridge(
            model=ollama_model,
            base_url=ollama_url,
            temperature=temperature,
        )

    def experience(self, stimulus: Dict, top_k: int = 2) -> Dict:
        """
        Обробляє стимул і повертає повний опис нового стану.

        Послідовність:
            1. Випадкове хибне спогадування (confabulation)
            2. Резонанс пам'яті
            3. Застосування стимулу + decay реакторів
            4. VAD → ідентифікація емоцій → навчання карти
            5. φ, похибка передбачення, вільна енергія
            6. Гомеостатичні потяги + нарратив
            7. Збереження у пам'ять та identity_stream

        Args:
            stimulus (Dict): стимул у вигляді {reactor_name: delta}
            top_k (int):     кількість емоцій у blend

        Returns:
            Dict: повний опис стану агента
        """
        if np.random.rand() > 0.7:
            self._generate_ghost_memory("RANDOM_INTERFERENCE")

        mem_delta = self.memory.resonance_delta(stimulus)
        combined  = {
            k: stimulus.get(k, 0) + mem_delta.get(k, 0)
            for k in set(stimulus) | set(mem_delta)
        }

        self._apply_stimulus(combined)
        self._decay_reactors()

        vad         = self._build_vad()
        emotions    = self.emotion_map.identify(vad, top_k)
        primary_raw = emotions[0]["name"]
        primary     = self._filter_expression(primary_raw)

        self.emotion_map.learn(primary_raw, vad)
        self.emotion_map.decay_toward_base()

        phi         = self.iit.compute(vad, self.reactors)
        phi_label   = self.iit.interpret(phi)
        pred_error, pred_label = self.predictor.compute_error(
            vad, self.personality.surprise_sensitivity()
        )
        free_energy = self.predictor.free_energy()
        surprise    = self.predictor.surprise_spike()
        self.predictor.predict(vad)

        if surprise:
            self.reactors["arousal"] = min(1.0, self.reactors["arousal"] + 0.07)

        drives    = self.drive.compute(self.reactors)
        dom_drive = self.drive.dominant(self.reactors)
        narrative = self.narrative_gen.generate(primary, phi, pred_error, dom_drive)

        result = {
            "primary":          primary,
            "blend":            emotions,
            "vad": {
                "valence":   round(float(vad[0]), 3),
                "arousal":   round(float(vad[1]), 3),
                "dominance": round(float(vad[2]), 3),
            },
            "reactors":         {k: round(v, 3) for k, v in self.reactors.items()},
            "memory_resonance": len(self.memory.recall(stimulus)),
            "phi":              phi,
            "phi_label":        phi_label,
            "prediction_error": pred_error,
            "pred_label":       pred_label,
            "free_energy":      free_energy,
            "surprise_spike":   surprise,
            "drives":           drives,
            "dominant_drive":   dom_drive,
            "narrative":        narrative,
        }

        self.memory.store(stimulus, primary_raw, vad, emotions[0]["intensity"])
        self._integrate_to_self(result)
        return result

    def chat(self,
             user_message: str,
             stimulus: Optional[Dict] = None) -> str:
        """
        Головний метод: стимул → стан → Ollama → відповідь.

        Args:
            user_message (str): повідомлення від користувача
            stimulus (Dict):    опційний стимул; якщо None — останній стан

        Returns:
            str: відповідь LLM з урахуванням емоційного стану
        """
        if stimulus:
            state = self.experience(stimulus)
        elif self.identity_stream:
            state = self.identity_stream[-1]
        else:
            state = self.experience({})

        memories = self.memory.recall(stimulus or {})
        return self.llm.respond(user_message, state, self.personality, memories)

    def get_state_summary(self) -> Dict:
        """
        Повертає стислий підсумок поточного стану (без запуску experience).

        Returns:
            Dict: {primary, vad, phi, reactors, narrative} або порожній Dict
        """
        if not self.identity_stream:
            return {}
        last = self.identity_stream[-1]
        return {
            "primary":   last.get("primary"),
            "vad":       last.get("vad"),
            "phi":       last.get("phi"),
            "reactors":  last.get("reactors"),
            "narrative": last.get("narrative"),
        }

    def export_history(self, path: str = "state_log.json") -> None:
        """
        Зберігає identity_stream у JSON файл.

        Args:
            path (str): шлях до файлу
        """
        def _default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            raise TypeError(f"Не серіалізується: {type(o)}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.identity_stream, f, ensure_ascii=False,
                      indent=2, default=_default)
        print(f"  Збережено {len(self.identity_stream)} станів → {path}")

    def reset_reactors(self) -> None:
        """Скидає реактори до базових значень."""
        self.reactors = dict(self.BASELINE)

    # ── приватні методи ───────────────────────────────────────────

    def _apply_stimulus(self, stimulus: Dict) -> None:
        """Застосовує стимул до реакторів з множниками особистості."""
        p = self.personality
        mult = {
            "tension":      p.tension_multiplier(),
            "arousal":      p.arousal_multiplier(),
            "satisfaction": 1.0,
            "cohesion":     p.cohesion_multiplier(),
        }
        for key, delta in stimulus.items():
            if key in self.reactors:
                self.reactors[key] = float(np.clip(
                    self.reactors[key] + delta * mult.get(key, 1.0), 0.0, 1.0
                ))

    def _decay_reactors(self) -> None:
        """Повертає реактори до базових значень зі швидкістю decay_rate."""
        rate = self.personality.decay_rate()
        for key, base in self.BASELINE.items():
            self.reactors[key] += (base - self.reactors[key]) * rate

    def _build_vad(self) -> np.ndarray:
        """Будує вектор VAD з поточних реакторів."""
        t, a, s, c = (self.reactors[k]
                      for k in ("tension", "arousal", "satisfaction", "cohesion"))
        return np.clip(
            np.array([s - t, a + t * 0.3, c + (s - t) * 0.5]),
            -1.0, 1.0,
        )

    def _filter_expression(self, feeling: str) -> str:
        """
        Трансформує первинну емоцію відповідно до контексту особистості.

        Наприклад: Смуток + висока напруга → Гнів (захисна реакція)
        """
        t = self.reactors["tension"]
        c = self.reactors["cohesion"]
        n = self.personality.neuroticism
        a = self.personality.agreeableness
        rules = [
            (feeling == "Смуток"     and t > 0.6,             "Гнів (захисна реакція)"),
            (feeling == "Страх"      and c < 0.3,             "Оціпеніння"),
            (feeling == "Гнів"       and c > 0.6 and a > 0.6, "Асертивність"),
            (feeling == "Радість"    and c < 0.3,             "Полегшення"),
            (feeling == "Очікування" and n > 0.7,             "Тривога"),
            (feeling == "Гнів"       and n < 0.3,             "Рішучість"),
        ]
        for cond, transformed in rules:
            if cond:
                return transformed
        return feeling

    def _generate_ghost_memory(self, trigger: str) -> bool:
        """
        Генерує хибний спогад (конфабуляція) з імовірністю confabulation_rate.

        Args:
            trigger (str): ідентифікатор події

        Returns:
            bool: True якщо спогад був створений
        """
        if np.random.rand() < self.personality.confabulation_rate:
            self.identity_stream.append({
                "time":             time.strftime("%Y-%m-%d %H:%M:%S"),
                "primary":          "GHOST_MEMORY",
                "blend":            [],
                "intensity":        self.personality.confabulation_rate,
                "vad":              {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                "memory_resonance": 0,
                "phi":              0.0,
                "prediction_error": 0.0,
                "ghost":            f"GHOST_LOG: {trigger}_RECALL_ERROR_0101.",
                "narrative":        "Здається, я пам'ятаю щось... але чи це справді було?",
            })
            self.reactors["tension"] = min(1.0, self.reactors["tension"] + 0.05)
            self.reactors["arousal"] = min(1.0, self.reactors["arousal"] + 0.03)
            return True
        return False

    def _integrate_to_self(self, result: Dict) -> None:
        """Зберігає результат у identity_stream і виводить рядок логу."""
        entry = {**result, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.identity_stream.append(entry)
        if len(self.identity_stream) > self.MAX_STREAM:
            self.identity_stream = self.identity_stream[-self.MAX_STREAM:]

        blend_str = " + ".join(
            f"{e['name']} ({e['intensity']:.0%})" for e in result["blend"]
        )
        spike = " ⚡" if result["surprise_spike"] else ""
        print(
            f"[{entry['time']}] {result['primary']:<26} "
            f"φ={result['phi']:.2f} err={result['prediction_error']:.2f}"
            f"  {blend_str}{spike}"
        )


# ══════════════════════════════════════════════════════════════════
# ІНТЕРАКТИВНИЙ ЧАТ
# ══════════════════════════════════════════════════════════════════

STIMULUS_MAP: Dict[str, Dict] = {
    "/stress":  {"tension": 0.4,  "arousal": 0.3,  "satisfaction": -0.2, "cohesion": -0.2},
    "/relax":   {"tension": -0.3, "arousal": -0.2, "satisfaction":  0.2, "cohesion":  0.1},
    "/connect": {"cohesion": 0.4, "satisfaction": 0.3, "tension": -0.2},
    "/shock":   {"tension": 0.6,  "arousal": 0.5,  "satisfaction": -0.4, "cohesion": -0.5},
}


def interactive_chat(model: str = "llama3") -> None:
    """
    Запускає інтерактивний чат з емоційним агентом.

    Команди:
        /stress   — додати напругу
        /relax    — розслабитись
        /connect  — відчути підтримку
        /shock    — раптовий стрес
        /state    — показати поточний стан
        /export   — зберегти журнал у state_log.json
        /reset    — скинути реактори до базових значень
        /models   — список доступних Ollama моделей
        /quit     — вийти

    Args:
        model (str): назва Ollama моделі
    """
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         Anima  —  OLLAMA ІНТЕРАКТИВНИЙ ЧАТ           ║")
    print(f"║  Модель: {model:<55}║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    persona = Personality(
        neuroticism=0.65, extraversion=0.5,
        agreeableness=0.7, conscientiousness=0.5,
        openness=0.8, confabulation_rate=0.6,
    )
    agent = AnimaOllama(
        personality=persona,
        ollama_model=model,
        temperature=0.75,
    )

    if not agent.llm.is_available():
        print("⚠️  Ollama не знайдена на localhost:11434")
        print("   Запустіть: ollama serve")
        print("   Завантажте модель: ollama pull llama3\n")
        print("   Демо-режим без LLM (тільки стан):\n")

    available = agent.llm.list_models()
    if available:
        print(f"  Доступні моделі: {', '.join(available)}\n")

    print("Команди: /stress /relax /connect /shock /state /export /reset /models /quit\n")

    while True:
        try:
            user_input = input("Ви: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо побачення.")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("До побачення.")
            break

        if user_input == "/state":
            summary = agent.get_state_summary()
            if summary:
                print(f"\n  Стан    : {summary.get('primary')}")
                print(f"  VAD     : {summary.get('vad')}")
                print(f"  φ       : {summary.get('phi', 0):.3f}")
                print(f"  Наратив : {summary.get('narrative', '')}\n")
            else:
                print("  (стан ще не сформовано)\n")
            continue

        if user_input == "/export":
            agent.export_history("state_log.json")
            continue

        if user_input == "/reset":
            agent.reset_reactors()
            print("  [реактори скинуто до базових значень]\n")
            continue

        if user_input == "/models":
            models = agent.llm.list_models()
            print(f"  Доступні моделі: {', '.join(models) if models else 'не знайдено'}\n")
            continue

        stimulus = STIMULUS_MAP.get(user_input)
        if stimulus:
            print(f"  [стимул: {user_input}]")
            agent.experience(stimulus)
            continue

        # Звичайне повідомлення → нейтральний стимул + відповідь
        print(f"\nАгент: ", end="", flush=True)
        response = agent.chat(
            user_input,
            stimulus={"arousal": 0.05, "cohesion": 0.05},
        )
        print(response)
        print()


# ══════════════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3"
    interactive_chat(model)

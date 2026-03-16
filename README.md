# Subjective Core — Ollama Integration

![Subjective Core Banner](./v4.jpg)

> An emotionally aware AI agent with a simulated inner life, running fully locally via [Ollama](https://ollama.com).

---

## What is this?

**Subjective Core** is a Python implementation of an AI agent with a modelled emotional system.  
The agent doesn't just generate text — it processes stimuli, forms memories, builds an internal narrative, and only then responds. Every LLM reply is shaped by the agent's current emotional state.

The project is grounded in several theoretical frameworks:

| Concept | Implementation |
|---|---|
| VAD (Valence-Arousal-Dominance) | 3D emotional state space |
| Big Five / OCEAN | Personality model |
| IIT (Integrated Information Theory) | Approximation of φ-integration |
| Predictive Processing (Friston) | Prediction error & free energy |
| Plutchik's Wheel | Eight basic emotions |
| Confabulation | Probabilistic false memory generation |

---

## Architecture

```
SubjectiveCoreOllama          ← main agent class
│
├── OllamaBridge              ← REST client for Ollama
├── Personality               ← Big Five traits + confabulation rate
│
├── AdaptiveEmotionMap        ← VAD → emotion map (learns over time)
├── AssociativeMemory         ← associative memory (cosine similarity)
├── IITModule                 ← φ computation (Tononi)
├── PredictiveProcessor       ← prediction error + free energy
├── HomeostaticDrive          ← homeostatic drives & needs
└── ExistentialNarrative      ← first-person inner narrative
```

### Stimulus Processing Loop

```
stimulus (Dict)
    ↓
[Confabulation?]   ← probabilistic false memories
    ↓
[Memory Resonance] ← past memories amplify current stimulus
    ↓
[Apply + Decay]    ← reactors: tension, arousal, satisfaction, cohesion
    ↓
[VAD Vector]       ← valence = satisfaction - tension, etc.
    ↓
[Emotion ID]       ← nearest emotion in VAD space
    ↓
[Filter]           ← contextual transform (Sadness+tension → Defensive Anger)
    ↓
[IIT φ]            ← entropy of whole vs. parts
    ↓
[Prediction Error] ← surprise level, free energy
    ↓
[Drives]           ← homeostatic needs
    ↓
[Narrative]        ← first-person internal narrative
    ↓
state Dict → OllamaBridge.build_system_prompt() → LLM → response
```

---

## Installation & Setup

### 1. Install Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download the installer from https://ollama.com
```

### 2. Pull a Model

```bash
ollama pull llama3        # recommended (~4.7 GB)
# or
ollama pull mistral       # (~4.1 GB)
ollama pull gemma2        # (~5.4 GB)
ollama pull phi3          # (~2.3 GB, fast)
ollama pull qwen2.5       # (~4.4 GB)
```

### 3. Install Python Dependencies

```bash
pip install requests numpy
```

### 4. Run

```bash
python subjective_core_ollama.py              # uses llama3 by default
python subjective_core_ollama.py mistral      # specify a different model
python subjective_core_ollama.py phi3         # lighter model
```

---

## Interactive Commands

| Command | Action |
|---|---|
| `(any text)` | Regular message — agent responds based on current emotional state |
| `/stress` | Increase tension and arousal |
| `/relax` | Reduce tension, calm down |
| `/connect` | Feel support and connection |
| `/shock` | Sudden strong stressor |
| `/state` | Display current emotional state |
| `/export` | Save state log → `state_log.json` |
| `/reset` | Reset all reactors to baseline |
| `/models` | List available Ollama models |
| `/quit` | Exit |

---

## Usage as a Library

```python
from subjective_core_ollama import SubjectiveCoreOllama, Personality

# Configure personality
persona = Personality(
    neuroticism=0.7,       # prone to anxiety
    extraversion=0.4,      # introverted
    agreeableness=0.8,     # very cooperative
    conscientiousness=0.6,
    openness=0.9,          # highly curious
    confabulation_rate=0.5,
)

agent = SubjectiveCoreOllama(
    personality=persona,
    ollama_model="mistral",
    temperature=0.8,
)

# Apply a stimulus manually
state = agent.experience({"tension": 0.3, "arousal": 0.2})
print(state["primary"])      # → "Anticipation"
print(state["narrative"])    # → "Something is about to happen..."
print(state["phi"])          # → 0.42

# Get an LLM response shaped by the current state
response = agent.chat(
    "How are you feeling?",
    stimulus={"cohesion": 0.1},
)
print(response)

# Export the session log
agent.export_history("my_session.json")
```

---

## State Dict Structure (`experience()` → Dict)

```python
{
    "primary":          "Anticipation",          # dominant emotion
    "blend": [
        {"name": "Anticipation", "intensity": 1.0},
        {"name": "Trust",        "intensity": 0.3},
    ],
    "vad": {
        "valence":   0.25,   # -1 (negative) ↔ +1 (positive)
        "arousal":   0.38,   # -1 (calm) ↔ +1 (excited)
        "dominance": 0.42,   # -1 (powerless) ↔ +1 (in control)
    },
    "reactors": {
        "tension":      0.22,
        "arousal":      0.31,
        "satisfaction": 0.48,
        "cohesion":     0.55,
    },
    "phi":              0.38,   # IIT φ-integration
    "phi_label":        "low integration",
    "prediction_error": 0.14,   # surprise level
    "pred_label":       "minor deviation",
    "free_energy":      0.09,
    "surprise_spike":   False,
    "dominant_drive":   None,   # or "cohesion" / "tension" / etc.
    "narrative":        "Something is about to happen. Attention directed forward.",
    "memory_resonance": 2,      # number of activated memories
}
```

---

## Reactors Explained

Four internal reactors define the agent's current state:

| Reactor | Baseline | Description |
|---|---|---|
| `tension` | 0.2 | Internal stress, strain |
| `arousal` | 0.2 | Activation, excitability |
| `satisfaction` | 0.5 | Contentment, calm |
| `cohesion` | 0.5 | Sense of connection, belonging |

VAD vector is derived from reactors:
```
valence   = satisfaction - tension
arousal   = arousal + tension * 0.3
dominance = cohesion + (satisfaction - tension) * 0.5
```

---

## Personality (Big Five)

| Parameter | Effect |
|---|---|
| `neuroticism` | Amplifies stress response; Sadness → Defensive Anger |
| `extraversion` | Amplifies arousal response |
| `agreeableness` | Amplifies cohesion; Anger → Assertiveness |
| `conscientiousness` | Speeds up emotional decay (return to baseline) |
| `openness` | Increases sensitivity to surprise |
| `confabulation_rate` | Probability of generating false memories |

---

## Emotion Filtering Rules

The agent does not always express its raw primary emotion. Context-dependent transforms apply:

| Raw Emotion | Condition | Expressed As |
|---|---|---|
| Sadness | High tension | Defensive Anger |
| Fear | Low cohesion | Numbness |
| Anger | High cohesion + high agreeableness | Assertiveness |
| Joy | Low cohesion | Relief |
| Anticipation | High neuroticism | Anxiety |
| Anger | Low neuroticism | Determination |

---

## How the System Prompt Works

The agent's emotional state is injected into the LLM system prompt as an `<inner_state>` XML block. The model is instructed to express the state through language only — never to quote numbers or tag names directly. Guidelines tie specific state values to tonal adjustments:

- High arousal → shorter, energetic sentences  
- Negative valence → cautious, withdrawn tone  
- High prediction error → express confusion naturally  
- Dominant drive `cohesion` → express need for connection  

The prompt is written in English for best cross-model compatibility, but the agent responds in whatever language the user writes in.

---

## Requirements

- Python 3.9+
- `numpy`
- `requests`
- Ollama running locally

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

### English Summary:
- **Personal & Academic Use:** You are free to use, modify, and build upon this code for non-commercial research and personal projects.
- **Commercial & Corporate Use:** **Strictly prohibited.** Any use by for-profit organizations, or integration into commercial products/services, requires a separate commercial license and written permission from the author.
- **Attribution:** You must give appropriate credit to the original author.

Copyright (c) 2026 [Stell]

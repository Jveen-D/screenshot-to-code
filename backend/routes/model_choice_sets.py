from llm import Llm

# Variant model sets are pinned to gpt-5.5.
VIDEO_VARIANT_MODELS = (
    Llm.GPT_5_5_HIGH,
)

# All API keys available.

# Image (Create)

ALL_KEYS_MODELS_DEFAULT = (
    Llm.GPT_5_5_HIGH,
)

# Text (Create)

ALL_KEYS_MODELS_TEXT_CREATE = (
    Llm.GPT_5_5_HIGH,
)

# Image + Text (Update)

ALL_KEYS_MODELS_UPDATE = (
    Llm.GPT_5_5_HIGH,
)

# Key subset fallbacks.
GEMINI_ANTHROPIC_MODELS = (
    Llm.GPT_5_5_HIGH,
)
GEMINI_OPENAI_MODELS = (
    Llm.GPT_5_5_HIGH,
)
OPENAI_ANTHROPIC_MODELS = (
    Llm.GPT_5_5_HIGH,
)
GEMINI_ONLY_MODELS = (
    Llm.GPT_5_5_HIGH,
)
ANTHROPIC_ONLY_MODELS = (
    Llm.GPT_5_5_HIGH,
)
OPENAI_ONLY_MODELS = (
    Llm.GPT_5_5_HIGH,
)

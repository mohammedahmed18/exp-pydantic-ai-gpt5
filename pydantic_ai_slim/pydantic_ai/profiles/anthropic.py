from __future__ import annotations as _annotations

from . import ModelProfile


def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model."""
    return _anthropic_model_profile

_ANTHROPIC_MODEL_PROFILE = ModelProfile(thinking_tags=('<thinking>', '</thinking>'))

_anthropic_model_profile = ModelProfile(thinking_tags=('<thinking>', '</thinking>'))

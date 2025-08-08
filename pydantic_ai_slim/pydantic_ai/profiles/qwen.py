from __future__ import annotations as _annotations

from . import InlineDefsJsonSchemaTransformer, ModelProfile


def qwen_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Qwen model."""
    return _profile

_QWEN_MODEL_PROFILE = ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)

_profile = ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)

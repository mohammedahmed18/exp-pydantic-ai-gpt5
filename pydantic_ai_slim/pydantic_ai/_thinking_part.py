from __future__ import annotations as _annotations

from pydantic_ai.messages import TextPart, ThinkingPart


def split_content_into_text_and_thinking(content: str, thinking_tags: tuple[str, str]) -> list[ThinkingPart | TextPart]:
    """Split a string into text and thinking parts.

    Some models don't return the thinking part as a separate part, but rather as a tag in the content.
    This function splits the content into text and thinking parts.
    """
    start_tag, end_tag = thinking_tags
    parts: list[ThinkingPart | TextPart] = []

    idx = 0
    start_tag_len = len(start_tag)
    end_tag_len = len(end_tag)

    while idx < len(content):
        start_index = content.find(start_tag, idx)
        if start_index == -1:
            if idx < len(content):
                parts.append(TextPart(content=content[idx:]))
            break
        if start_index > idx:
            parts.append(TextPart(content=content[idx:start_index]))
        end_index = content.find(end_tag, start_index + start_tag_len)
        if end_index == -1:
            parts.append(TextPart(content=content[start_index + start_tag_len:]))
            break
        parts.append(ThinkingPart(content=content[start_index + start_tag_len:end_index]))
        idx = end_index + end_tag_len

    return parts

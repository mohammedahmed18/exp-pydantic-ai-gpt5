from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from typing import Any
from xml.etree import ElementTree

from pydantic import BaseModel

__all__ = ('format_as_xml',)


def format_as_xml(
    obj: Any,
    root_tag: str | None = None,
    item_tag: str = 'item',
    none_str: str = 'null',
    indent: str | None = '  ',
) -> str:
    """Format a Python object as XML.

    This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML,
    rather than JSON etc.

    Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `Mapping`,
    `Iterable`, `dataclass`, and `BaseModel`.

    Args:
        obj: Python Object to serialize to XML.
        root_tag: Outer tag to wrap the XML in, use `None` to omit the outer tag.
        item_tag: Tag to use for each item in an iterable (e.g. list), this is overridden by the class name
            for dataclasses and Pydantic models.
        none_str: String to use for `None` values.
        indent: Indentation string to use for pretty printing.

    Returns:
        XML representation of the object.

    Example:
    ```python {title="format_as_xml_example.py" lint="skip"}
    from pydantic_ai import format_as_xml

    print(format_as_xml({'name': 'John', 'height': 6, 'weight': 200}, root_tag='user'))
    '''
    <user>
      <name>John</name>
      <height>6</height>
      <weight>200</weight>
    </user>
    '''
    ```
    """
    tx = _ToXml(item_tag=item_tag, none_str=none_str)
    el = tx.to_xml(obj, root_tag)
    if root_tag is None and el.text is None:
        if indent is not None:
            join = '\n'
        else:
            join = ''
        return join.join(_rootless_xml_elements(el, indent))
    else:
        if indent is not None:
            ElementTree.indent(el, space=indent)
        return ElementTree.tostring(el, encoding='unicode')


@dataclass
class _ToXml:
    item_tag: str
    none_str: str

    def to_xml(self, value: Any, tag: str | None) -> ElementTree.Element:
        # Fast-path for None
        if value is None:
            element = ElementTree.Element(self.item_tag if tag is None else tag)
            element.text = self.none_str
            return element

        # Fast-path for primitives and str/bytes
        v_type = type(value)
        if v_type is str:
            element = ElementTree.Element(self.item_tag if tag is None else tag)
            element.text = value
            return element

        if v_type is bytes or v_type is bytearray:
            element = ElementTree.Element(self.item_tag if tag is None else tag)
            element.text = value.decode(errors='ignore')
            return element

        if v_type is bool or v_type is int or v_type is float:
            element = ElementTree.Element(self.item_tag if tag is None else tag)
            element.text = str(value)
            return element

        if isinstance(value, date):
            element = ElementTree.Element(self.item_tag if tag is None else tag)
            element.text = value.isoformat()
            return element

        # Mapping
        if isinstance(value, Mapping):
            element = ElementTree.Element(self.item_tag if tag is None else tag)
            self._mapping_to_xml(element, value)
            return element

        # Dataclass
        if is_dataclass(value) and not isinstance(value, type):
            dc_tag = value.__class__.__name__ if tag is None else tag
            element = ElementTree.Element(dc_tag)
            # Use fields instead of asdict for efficiency and to preserve nested structure
            for f in fields(value):
                v = getattr(value, f.name)
                child = self.to_xml(v, f.name)
                element.append(child)
            return element

        # BaseModel
        if isinstance(value, BaseModel):
            bm_tag = value.__class__.__name__ if tag is None else tag
            element = ElementTree.Element(bm_tag)
            # .model_dump(mode='python') avoids serialization overhead
            self._mapping_to_xml(element, value.model_dump(mode='python'))
            return element

        # Iterable (avoid treating dicts/strings as iterables!)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
            element = ElementTree.Element(self.item_tag if tag is None else tag)
            app = element.append  # localize for speed
            for item in value:
                app(self.to_xml(item, None))
            return element

        raise TypeError(f'Unsupported type for XML formatting: {type(value)}')

    def _mapping_to_xml(self, element: ElementTree.Element, mapping: Mapping[Any, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(key, int):
                key = str(key)
            elif not isinstance(key, str):
                raise TypeError(f'Unsupported key type for XML formatting: {type(key)}, only str and int are allowed')
            element.append(self.to_xml(value, key))


def _rootless_xml_elements(root: ElementTree.Element, indent: str | None) -> Iterator[str]:
    # No optimization here except localizing functions (negligible in most real cases).
    for sub_element in root:
        if indent is not None:
            ElementTree.indent(sub_element, space=indent)
        yield ElementTree.tostring(sub_element, encoding='unicode')

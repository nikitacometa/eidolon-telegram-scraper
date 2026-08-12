"""Provider-contract gates for every Pydantic structured-output schema."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from pipeline.indexer import BatchExtractionResult, ExtractionResult
from pipeline.models import ModelClassification
from pipeline.summarizer import DigestResult

STRUCTURED_OUTPUT_MODELS: tuple[type[BaseModel], ...] = (
    BatchExtractionResult,
    ExtractionResult,
    ModelClassification,
    DigestResult,
)

_SUPPORTED_KEYWORDS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "anyOf",
        "const",
        "description",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "items",
        "maximum",
        "maxItems",
        "maxLength",
        "minimum",
        "minItems",
        "minLength",
        "multipleOf",
        "pattern",
        "properties",
        "required",
        "title",
        "type",
    }
)
_STRING_KEYWORDS = frozenset({"format", "maxLength", "minLength", "pattern"})
_NUMBER_KEYWORDS = frozenset(
    {"exclusiveMaximum", "exclusiveMinimum", "maximum", "minimum", "multipleOf"}
)
_ARRAY_KEYWORDS = frozenset({"maxItems", "minItems"})


def _schema_nodes(
    node: Mapping[str, Any], path: str = "$"
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    yield path, node
    definitions = node.get("$defs", {})
    assert isinstance(definitions, dict), f"{path}.$defs must be an object"
    for name, definition in definitions.items():
        assert isinstance(definition, dict), f"{path}.$defs.{name} must be a schema"
        yield from _schema_nodes(definition, f"{path}.$defs.{name}")
    properties = node.get("properties", {})
    assert isinstance(properties, dict), f"{path}.properties must be an object"
    for name, property_schema in properties.items():
        assert isinstance(property_schema, dict), f"{path}.properties.{name} must be a schema"
        yield from _schema_nodes(property_schema, f"{path}.properties.{name}")
    items = node.get("items")
    if items is not None:
        assert isinstance(items, dict), f"{path}.items must be a schema"
        yield from _schema_nodes(items, f"{path}.items")
    alternatives = node.get("anyOf", [])
    assert isinstance(alternatives, list), f"{path}.anyOf must be an array"
    for index, alternative in enumerate(alternatives):
        assert isinstance(alternative, dict), f"{path}.anyOf[{index}] must be a schema"
        yield from _schema_nodes(alternative, f"{path}.anyOf[{index}]")


def _assert_openai_schema_contract(model: type[BaseModel]) -> None:
    schema = model.model_json_schema()
    assert schema.get("type") == "object", "Structured Outputs requires an object root"
    assert "anyOf" not in schema, "Structured Outputs forbids anyOf at the root"

    for path, node in _schema_nodes(schema):
        unsupported = set(node) - _SUPPORTED_KEYWORDS
        assert not unsupported, f"{path} has unsupported JSON Schema keywords: {unsupported}"

        node_types = node.get("type")
        types = {node_types} if isinstance(node_types, str) else set(node_types or [])
        assert not (_STRING_KEYWORDS & node.keys()) or "string" in types, (
            f"{path} puts string constraints on {node_types!r}"
        )
        assert not (_NUMBER_KEYWORDS & node.keys()) or types & {"integer", "number"}, (
            f"{path} puts numeric constraints on {node_types!r}"
        )
        assert not (_ARRAY_KEYWORDS & node.keys()) or "array" in types, (
            f"{path} puts array constraints on {node_types!r}"
        )

        if "properties" in node:
            properties = node["properties"]
            assert isinstance(properties, dict)
            assert node.get("additionalProperties") is False, (
                f"{path} must set additionalProperties=false"
            )
            assert set(node.get("required", [])) == set(properties), (
                f"{path} must require every property"
            )


@pytest.mark.parametrize("model", STRUCTURED_OUTPUT_MODELS, ids=lambda model: model.__name__)
def test_structured_output_schema_matches_openai_contract(model: type[BaseModel]) -> None:
    _assert_openai_schema_contract(model)


def test_every_parse_response_model_is_registered_for_schema_validation() -> None:
    """A new response model must not bypass the provider-contract gate."""
    used_models: set[str] = set()
    for path in Path("pipeline").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg == "response_format" and isinstance(keyword.value, ast.Name):
                    used_models.add(keyword.value.id)

    registered = {model.__name__ for model in STRUCTURED_OUTPUT_MODELS}
    assert used_models == registered

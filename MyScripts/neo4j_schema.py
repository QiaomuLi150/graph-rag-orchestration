from typing import Any, Literal, Optional

import neo4j


SchemaMode = Literal["full_schema", "core_schema"]

NODE_PROPERTIES_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "node"
WITH label AS nodeLabels, collect({property: property, type: type}) AS properties
RETURN {labels: nodeLabels, properties: properties} AS output
"""

REL_PROPERTIES_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
WITH label AS relType, collect({property: property, type: type}) AS properties
RETURN {type: relType, properties: properties} AS output
"""

REL_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE type = "RELATIONSHIP" AND elementType = "node"
UNWIND other AS other_node
RETURN {start: label, type: property, end: toString(other_node)} AS output
"""

CORE_REL_TYPE = "SUMMARIZED_RELATIONSHIP"
EXCLUDED_CORE_NODE_LABELS = {"__Entity__"}


def _format_props(props: list[dict[str, Any]]) -> str:
    return ", ".join([f"{prop['property']}: {prop['type']}" for prop in props])


def _apply_schema_mode(
    structured_schema: dict[str, Any],
    mode: SchemaMode = "full_schema",
) -> dict[str, Any]:
    if mode == "full_schema":
        return structured_schema

    if mode != "core_schema":
        raise ValueError(f"Unsupported schema mode: {mode}")

    summarized_relationships = [
        rel for rel in structured_schema["relationships"] if rel["type"] == CORE_REL_TYPE
    ]

    related_labels = {
        rel["start"] for rel in summarized_relationships
    } | {
        rel["end"] for rel in summarized_relationships
    }

    related_labels = {
        label for label in related_labels if label not in EXCLUDED_CORE_NODE_LABELS
    }

    filtered_relationships = [
        rel
        for rel in summarized_relationships
        if rel["start"] in related_labels and rel["end"] in related_labels
    ]

    filtered_node_props = {
        label: props
        for label, props in structured_schema["node_props"].items()
        if label in related_labels
    }

    filtered_rel_props = {
        rel_type: props
        for rel_type, props in structured_schema["rel_props"].items()
        if rel_type == CORE_REL_TYPE
    }

    return {
        "node_props": filtered_node_props,
        "rel_props": filtered_rel_props,
        "relationships": filtered_relationships,
    }


def get_structured_schema(
    driver: neo4j.Driver,
    mode: SchemaMode = "full_schema",
    database_: Optional[str] = None,
) -> dict[str, Any]:
    node_labels_response = driver.execute_query(
        NODE_PROPERTIES_QUERY, database_=database_
    )
    node_properties = [data["output"] for data in [r.data() for r in node_labels_response.records]]

    rel_properties_query_response = driver.execute_query(
        REL_PROPERTIES_QUERY, database_=database_
    )
    rel_properties = [
        data["output"] for data in [r.data() for r in rel_properties_query_response.records]
    ]

    rel_query_response = driver.execute_query(REL_QUERY, database_=database_)
    relationships = [data["output"] for data in [r.data() for r in rel_query_response.records]]

    structured_schema = {
        "node_props": {el["labels"]: el["properties"] for el in node_properties},
        "rel_props": {el["type"]: el["properties"] for el in rel_properties},
        "relationships": relationships,
    }

    return _apply_schema_mode(structured_schema, mode=mode)


def get_schema(
    driver: neo4j.Driver,
    mode: SchemaMode = "full_schema",
    database_: Optional[str] = None,
) -> str:
    structured_schema = get_structured_schema(driver, mode=mode, database_=database_)

    formatted_node_props = [
        f"{label} {{{_format_props(props)}}}"
        for label, props in structured_schema["node_props"].items()
    ]
    formatted_rel_props = [
        f"{rel_type} {{{_format_props(props)}}}"
        for rel_type, props in structured_schema["rel_props"].items()
    ]
    formatted_rels = [
        f"(:{element['start']})-[:{element['type']}]->(:{element['end']})"
        for element in structured_schema["relationships"]
    ]

    return "\n".join(
        [
            "Node properties:",
            "\n".join(formatted_node_props),
            "Relationship properties:",
            "\n".join(formatted_rel_props),
            "The relationships:",
            "\n".join(formatted_rels),
        ]
    )

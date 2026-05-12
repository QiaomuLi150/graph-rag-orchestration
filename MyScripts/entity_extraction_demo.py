from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

import extraction_tools
from runtime import chat, chunk_text, ensure_neo4j_driver, read_text


ENTITY_TYPE_DEFINITIONS = {
    "AGENT": "An individual acting entity, such as a person, deity, ruler, named fictional being, or other named actor.",
    "ORGANIZATION": "A collective or institutional entity such as a company, government, army, religious order, team, or group.",
    "LOCATION": "A place or geographic/political region such as a city, country, sea, island, kingdom, or physical setting.",
    "EVENT": "A time-bounded occurrence such as a war, ceremony, ritual occurrence, meeting, battle, journey, or incident.",
    "DOCUMENT": "A text-bearing information object such as a book, report, contract, letter, paper, decree, or written record.",
    "ARTIFACT": "A man-made object, tool, weapon, vessel, or physical item that is not primarily a document or system.",
    "SYSTEM": "A technical, organizational, or functional system made of interacting components, such as a platform, pipeline, architecture, network, or process framework.",
    "CONCEPT": "An abstract idea, doctrine, role, title, policy, method, metric, belief, or category.",
    "OTHER": "Use only when none of the other entity types fit clearly.",
}

ENTITY_TYPES = list(ENTITY_TYPE_DEFINITIONS.keys())


class EntityType(str):
    pass


def normalize_entity_type_label(value: str) -> str:
    return str(value).strip().upper().replace(" ", "_").replace("-", "_")


def format_entity_type_definitions() -> str:
    return "\n".join(
        [f"- {entity_type}: {definition}" for entity_type, definition in ENTITY_TYPE_DEFINITIONS.items()]
    )


def build_extraction_prompt(text: str) -> str:
    base_prompt = extraction_tools.create_extraction_prompt(ENTITY_TYPES, text)
    definitions_block = format_entity_type_definitions()

    return f"""
{base_prompt}

Use the following entity type definitions when classifying entities:

{definitions_block}

Classification rules:
- Choose exactly one entity_type from the allowed list.
- Prefer the most specific allowed type that still matches the definition.
- Use OTHER only when none of the other types fit clearly.
- Be consistent across chunks.
- Do not invent new entity types.
"""


class EntityRecord(BaseModel):
    entity_name: str = Field(..., min_length=1)
    entity_type: str
    entity_description: str = Field(..., min_length=1)

    @field_validator("entity_name")
    @classmethod
    def normalize_entity_name(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("entity_type", mode="before")
    @classmethod
    def normalize_entity_type(cls, v):
        return normalize_entity_type_label(v)


class RelationshipRecord(BaseModel):
    source_entity: str = Field(..., min_length=1)
    target_entity: str = Field(..., min_length=1)
    relationship_description: str = Field(..., min_length=1)
    relationship_strength: float = Field(..., ge=0, le=10)

    @field_validator("source_entity", "target_entity")
    @classmethod
    def normalize_related_names(cls, v: str) -> str:
        return v.strip().upper()


MAX_RETRIES = 2
REJECTION_RATIO_THRESHOLD = 0.3


def validate_records(raw_nodes, raw_relationships):
    valid_nodes = []
    valid_relationships = []
    rejected = []

    for node in raw_nodes:
        try:
            validated = EntityRecord.model_validate(node)
            valid_nodes.append(
                {
                    "entity_name": validated.entity_name,
                    "entity_type": validated.entity_type,
                    "entity_description": validated.entity_description,
                }
            )
        except Exception as e:
            rejected.append({"kind": "entity", "record": node, "error": str(e)})

    for rel in raw_relationships:
        try:
            validated = RelationshipRecord.model_validate(rel)
            valid_relationships.append(
                {
                    "source_entity": validated.source_entity,
                    "target_entity": validated.target_entity,
                    "relationship_description": validated.relationship_description,
                    "relationship_strength": validated.relationship_strength,
                }
            )
        except Exception as e:
            rejected.append({"kind": "relationship", "record": rel, "error": str(e)})

    return valid_nodes, valid_relationships, rejected


def should_retry(valid_nodes, valid_relationships, rejected):
    total = len(valid_nodes) + len(valid_relationships) + len(rejected)
    if len(valid_nodes) == 0 and len(valid_relationships) == 0:
        return True
    if total == 0:
        return True
    rejection_ratio = len(rejected) / total
    return rejection_ratio > REJECTION_RATIO_THRESHOLD


def build_retry_prompt(text: str, errors: list) -> str:
    original_prompt = build_extraction_prompt(text)
    error_summary = "\n".join([f"- {item['kind']}: {item['error']}" for item in errors[:10]])

    return f"""
{original_prompt}

Your previous output failed schema validation.

Validation issues:
{error_summary}

Please regenerate the extraction and strictly follow the required schema.
Use only the allowed entity types.
Do not omit required fields.
Do not return malformed entity or relationship tuples.
"""


def extract_entities(text: str):
    prompt = build_extraction_prompt(text)
    final_valid_nodes = []
    final_valid_relationships = []
    final_rejected = []

    for attempt in range(MAX_RETRIES + 1):
        output = chat([{"role": "user", "content": prompt}], model="gpt-5-nano")
        raw_nodes, raw_relationships = extraction_tools.parse_extraction_output(output)

        valid_nodes, valid_relationships, rejected = validate_records(raw_nodes, raw_relationships)
        print(
            f"Attempt {attempt + 1}: {len(valid_nodes)} valid nodes, "
            f"{len(valid_relationships)} valid relationships, {len(rejected)} rejected"
        )

        final_valid_nodes = valid_nodes
        final_valid_relationships = valid_relationships
        final_rejected = rejected

        if not should_retry(valid_nodes, valid_relationships, rejected):
            return valid_nodes, valid_relationships

        prompt = build_retry_prompt(text, rejected)

    print(f"Final fallback after retries. Rejected {len(final_rejected)} records.")
    return final_valid_nodes, final_valid_relationships


def load_demo_text() -> str:
    sample_path = Path(__file__).resolve().parent / "sample_data" / "entity_extraction_sample.txt"
    return read_text(sample_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Entity extraction demo")
    parser.add_argument(
        "--database",
        default=None,
        help="Optional Neo4j database name to import into",
    )
    args = parser.parse_args()

    driver = ensure_neo4j_driver()
    text = load_demo_text()
    chunks = chunk_text(text, 1000, 40)

    for chunk_i, chunk in enumerate(tqdm(chunks, desc="Processing demo chunks")):
        nodes, relationships = extract_entities(chunk)
        print(f"Chunk {chunk_i}: {len(nodes)} valid nodes, {len(relationships)} valid relationships")

        driver.execute_query(
            extraction_tools.import_nodes_query,
            data=nodes,
            book_id=0,
            text=chunk,
            chunk_id=chunk_i,
            database_=args.database,
        )
        driver.execute_query(
            extraction_tools.import_relationships_query,
            data=relationships,
            database_=args.database,
        )


if __name__ == "__main__":
    main()

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

from neo4j_schema import get_schema, get_structured_schema
from runtime import chat, ensure_neo4j_driver, neo4j_driver


prompt_template = {
    "static": {
        "instructions": """
Instructions:
Generate Cypher statement to query a graph database to get the data to answer the user question below.

Format instructions:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to 
construct a Cypher statement.
Do not include any text except the generated Cypher statement.
ONLY RESPOND WITH CYPHER, NO CODEBLOCKS.
Make sure to name RETURN variables as requested in the user question.

Entity name normalization:
All entity names stored in the graph are UPPERCASE.
When matching against an entity name property such as `name`, normalize the user-provided name to uppercase.
Prefer exact matches on uppercase names.
""",
    },
    "dynamic": {
        "schema": """
Graph Database Schema:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided in the schema.
{}
""",
        "terminology": """
Terminology mapping:
This section is helpful to map terminology between the user question and the graph database schema.
{}
""",
        "examples": """
Examples:
The following examples provide useful patterns for querying the graph database.
{}
""",
        "question": """
User question: {}
""",
    },
}


class Text2Cypher:
    def __init__(
        self,
        driver=None,
        schema_mode: str = "full_schema",
        database_: Optional[str] = None,
    ):
        self.driver = driver or neo4j_driver or ensure_neo4j_driver()
        self.schema_mode = schema_mode
        self.database_ = database_
        self.dynamic_sections = {}
        self.required_sections = ["question"]
        self.prompt_template = prompt_template

        schema_string = get_schema(self.driver, mode=schema_mode, database_=database_)
        self.set_prompt_section("schema", schema_string)

    def set_prompt_section(
        self,
        section: Literal["terminology", "examples", "schema", "question"],
        value: str,
    ):
        self.dynamic_sections[section] = value

    def get_full_prompt(self):
        prompt = self.prompt_template["static"]["instructions"]
        for section in self.prompt_template["dynamic"]:
            if section in self.dynamic_sections:
                prompt += self.prompt_template["dynamic"][section].format(
                    self.dynamic_sections[section]
                )
        return prompt

    def generate_cypher(self):
        for section in self.required_sections:
            if section not in self.dynamic_sections:
                raise ValueError(
                    f"Section {section} is required to generate a prompt. Use set_prompt_section to set it."
                )
        prompt = self.get_full_prompt()
        return chat(messages=[{"role": "user", "content": prompt}])


class NodeRule(BaseModel):
    label: str
    title: str
    aliases: list[str] = Field(default_factory=list)


class RelationshipRule(BaseModel):
    type: str
    title: str
    aliases: list[str] = Field(default_factory=list)


class ExampleItem(BaseModel):
    question: str
    cypher: str


class PromptPack(BaseModel):
    node_rules: list[NodeRule]
    relationship_rules: list[RelationshipRule]
    examples: list[ExampleItem]


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model did not return JSON:\n{text}")

    return json.loads(text[start : end + 1])


def _validate_cypher(driver, cypher: str, database_: Optional[str] = None):
    try:
        driver.execute_query(f"EXPLAIN {cypher}", database_=database_)
        return True, None
    except Exception as e:
        return False, str(e)


def _normalize_value(value, text_limit=220):
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) == 0:
            return []
        if all(isinstance(x, (int, float)) for x in value[:10]):
            return f"<numeric_list len={len(value)}>"
        return [str(x)[:text_limit] for x in value[:3]]
    if isinstance(value, str):
        return value[:text_limit]
    if isinstance(value, (int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)[:text_limit]


def _sample_nodes_for_label(
    driver,
    label: str,
    properties: list[dict],
    limit: int = 3,
    database_: Optional[str] = None,
):
    prop_names = [p["property"] for p in properties]
    if not prop_names:
        return []

    query = f"""
    MATCH (n:`{label}`)
    WITH n, rand() AS r
    ORDER BY r
    LIMIT $limit
    RETURN properties(n) AS props
    """
    result = driver.execute_query(query, {"limit": limit}, database_=database_)
    rows = [r.data()["props"] for r in result.records]

    samples = []
    for row in rows:
        cleaned = {}
        for prop in prop_names:
            cleaned[prop] = _normalize_value(row.get(prop))
        samples.append(cleaned)
    return samples


def _sample_schema_semantics(
    driver,
    structured_schema: dict,
    per_label: int = 3,
    database_: Optional[str] = None,
):
    node_samples = {}
    for label, props in structured_schema["node_props"].items():
        node_samples[label] = {
            "properties": props,
            "samples": _sample_nodes_for_label(
                driver,
                label,
                props,
                limit=per_label,
                database_=database_,
            ),
        }
    return node_samples


def _join_aliases(aliases: list[str]) -> str:
    aliases = [a.strip() for a in aliases if a and a.strip()]
    aliases = list(dict.fromkeys(aliases))
    if not aliases:
        return "this concept"
    if len(aliases) == 1:
        return aliases[0]
    if len(aliases) == 2:
        return f"{aliases[0]} or {aliases[1]}"
    return ", ".join(aliases[:-1]) + f", or {aliases[-1]}"


def _render_terminology_string(pack: PromptPack) -> str:
    lines = []
    for rule in pack.node_rules:
        alias_text = _join_aliases(rule.aliases)
        lines.append(
            f"{rule.title}: When a user asks about {alias_text}, they are referring to a node with the label '{rule.label}'."
        )
    return "\n".join(lines)


def _render_examples_string(pack: PromptPack) -> str:
    return "\n\n".join(f"Question: {ex.question}\nCypher: {ex.cypher}" for ex in pack.examples)


def _check_schema_coverage(pack: PromptPack, structured_schema: dict):
    expected_labels = set(structured_schema["node_props"].keys())
    actual_labels = {r.label for r in pack.node_rules}

    expected_rels = set(structured_schema["rel_props"].keys())
    actual_rels = {r.type for r in pack.relationship_rules}

    missing_labels = expected_labels - actual_labels
    missing_rels = expected_rels - actual_rels

    if missing_labels:
        raise ValueError(f"Missing node labels: {sorted(missing_labels)}")
    if missing_rels:
        raise ValueError(f"Missing relationship types: {sorted(missing_rels)}")


def generate_prompt_sections(
    driver=None,
    max_examples: int = 6,
    max_retries: int = 2,
    per_label_samples: int = 3,
    database_: Optional[str] = None,
):
    driver = driver or neo4j_driver or ensure_neo4j_driver()
    structured_schema = get_structured_schema(driver, "core_schema", database_=database_)
    schema_string = get_schema(driver, "core_schema", database_=database_)
    node_samples = _sample_schema_semantics(
        driver,
        structured_schema,
        per_label=per_label_samples,
        database_=database_,
    )

    schema_json = json.dumps(structured_schema, indent=2)
    sample_json = json.dumps(node_samples, indent=2)
    node_labels = list(structured_schema["node_props"].keys())
    rel_types = list(structured_schema["rel_props"].keys())

    system_prompt = f"""
You generate prompt sections for a Text2Cypher system.

You are given:
1. a Neo4j schema
2. a small sample of real nodes for each label

Use the samples only to understand what each label represents in this specific database.

Your task is to infer practical, user-facing terminology that helps a Text2Cypher system map natural-language questions to the correct node labels and relationship types.

Return JSON ONLY with this exact shape:

{{
  "node_rules": [
    {{
      "label": "PERSON",
      "title": "Persons",
      "aliases": ["person", "people", "human", "character"]
    }}
  ],
  "relationship_rules": [
    {{
      "type": "RELATIONSHIP",
      "title": "Entity relationships",
      "aliases": ["relation", "connection", "linked entity"]
    }}
  ],
  "examples": [
    {{
      "question": "Which people are most mentioned in a book?",
      "cypher": "MATCH ..."
    }}
  ]
}}

Rules for node_rules:
- You MUST include one node_rule for EVERY node label in the schema.
- Each node_rule must contain:
  - the exact label
  - a short plural title
  - 6 to 10 common user-facing aliases
- Keep aliases short and natural.
- Do NOT mention properties.
- Do NOT explain graph structure.
- Do NOT write long descriptions.
- Do NOT collapse separate labels into one generic bucket.
- Use the samples to infer what the label means in practice.

Alias quality requirements:
- Aliases should reflect how a user would naturally refer to that label in questions.
- Infer aliases from the actual semantic background shown in the samples, not just from the label text alone.
- Use the samples to detect the real concept behind the label: role, category, domain meaning, and likely user wording.
- Prefer aliases that are useful for retrieval and disambiguation in this specific database.
- Include a mix of:
  - broad everyday user words
  - domain-grounded words clearly supported by the samples
- If the samples strongly suggest a domain-specific meaning, include a few domain-aware aliases.
- If multiple samples suggest recurring roles or themes, use those to make aliases more insightful.
- Do not invent niche aliases that are not supported by the samples.
- Do not use aliases that are so broad they could easily refer to many other labels.
- Avoid trivial aliases that merely restate the label with no added semantic value.

Interpretation guidance:
- Look across multiple sampled nodes for recurring patterns in names, summaries, and descriptions.
- Infer what users are likely to ask for when they mean this label.
- If a label represents actors, places, groups, artifacts, concepts, events, or documents in a domain-specific way, capture that with concise aliases.
- If the samples imply mythology, literature, enterprise, law, science, or another domain, reflect that only when clearly supported by the samples.
- Stay cautious: use the samples to sharpen the aliases, not to overfit.

Entity name normalization:
- All entity names stored in the graph are UPPERCASE.
- When matching against an entity name property such as `name`, normalize the user-provided name to uppercase.
- Prefer exact matches on uppercase names.

Good style:
- Persons -> ["person", "people", "human", "character"]
- Movies -> ["movie", "film"]
- Organizations in a mythic corpus -> ["group", "house", "kingdom", "order"]
- Agents in a mythic corpus -> ["person", "character", "hero", "ruler"]

Bad style:
- "core concepts extracted from text"
- "content container with metadata"
- "key properties include ..."
- aliases that simply mirror the label without adding meaning
- aliases that are too broad and fit every label

Rules for relationship_rules:
- Keep them short as well.
- They are mainly for internal example generation, not for verbose terminology text.
- Use the schema and samples to infer short, natural relationship wording when possible.
- Do not make them overly generic if the graph clearly represents a more specific kind of linkage.

Rules for examples:
- Examples must use only schema-valid labels, relationships, and properties.
- Examples should reflect realistic user questions for this specific database.
- Use the samples to make the questions domain-aware, but keep them natural and concise.
- Generate at most {max_examples} examples.
"""

    user_prompt = f"""
Schema JSON:
{schema_json}

Node samples:
{sample_json}

Expected node labels:
{node_labels}

Expected relationship types:
{rel_types}

Human-readable schema:
{schema_string}
"""

    raw = chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    pack = PromptPack.model_validate(_extract_json(raw))

    for _ in range(max_retries + 1):
        try:
            _check_schema_coverage(pack, structured_schema)
        except Exception as e:
            repair_prompt = f"""
The previous output did not fully cover the schema.

Schema JSON:
{schema_json}

Node samples:
{sample_json}

Current output:
{pack.model_dump_json(indent=2)}

Problem:
{str(e)}

Return the FULL corrected JSON again.
"""
            raw = chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": repair_prompt},
                ]
            )
            pack = PromptPack.model_validate(_extract_json(raw))
            continue

        invalid = []
        for ex in pack.examples:
            ok, err = _validate_cypher(driver, ex.cypher, database_=database_)
            if not ok:
                invalid.append({"question": ex.question, "cypher": ex.cypher, "error": err})

        if not invalid:
            break

        repair_prompt = f"""
Some examples are invalid.

Schema JSON:
{schema_json}

Node samples:
{sample_json}

Invalid examples:
{json.dumps(invalid, indent=2)}

Current output:
{pack.model_dump_json(indent=2)}

Return the FULL corrected JSON again.
"""
        raw = chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": repair_prompt},
            ]
        )
        pack = PromptPack.model_validate(_extract_json(raw))

    _check_schema_coverage(pack, structured_schema)
    for ex in pack.examples:
        ok, err = _validate_cypher(driver, ex.cypher, database_=database_)
        if not ok:
            raise ValueError(f"Invalid example after retries: {ex.question} -> {err}")

    return {
        "schema_string": schema_string,
        "terminology_string": _render_terminology_string(pack),
        "examples_string": _render_examples_string(pack),
        "node_samples": node_samples,
        "pack": pack,
    }

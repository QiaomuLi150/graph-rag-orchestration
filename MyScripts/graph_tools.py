from typing import Optional

from cypher_generation import Text2Cypher
from runtime import ensure_neo4j_driver, neo4j_driver


answer_given_description = {
    "type": "function",
    "function": {
        "name": "answer_given",
        "description": (
            "If the conversation already contains a complete answer to the question, "
            "use this tool to extract it. If the user engages in small talk, remind "
            "them that you can only answer questions grounded in the current graph database."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Respond directly with the answer",
                }
            },
            "required": ["answer"],
        },
    },
}


def answer_given(answer: str):
    return answer


text2cypher_description = {
    "type": "function",
    "function": {
        "name": "text2cypher",
        "description": (
            "Query the database with a user question. "
            "When the direct entity lookup tool does not fit, fallback to this one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user question to find the answer for",
                }
            },
            "required": ["question"],
        },
    },
}


def text2cypher(question: str, database_: Optional[str] = None):
    driver = neo4j_driver or ensure_neo4j_driver()
    t2c = Text2Cypher(driver, database_=database_)
    t2c.set_prompt_section("question", question)
    cypher = t2c.generate_cypher()
    try:
        records, _, _ = driver.execute_query(cypher, database_=database_)
        return [record.data() for record in records]
    except Exception as e:
        return [f"{cypher} cause an error: {e}"]


entity_info_by_name_description = {
    "type": "function",
    "function": {
        "name": "entity_info_by_name",
        "description": (
            "Get information about an entity by providing its name. "
            "This unified direct-lookup tool works for any entity type in the graph and "
            "returns the matched entity together with its SUMMARIZED_RELATIONSHIP connections."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": (
                        "The entity name to search for. Entity names in the graph are stored in UPPERCASE "
                        "and will be normalized to uppercase before matching."
                    ),
                },
                "entity_label": {
                    "type": "string",
                    "description": (
                        "Optional entity label to narrow the search, such as PERSON, LOCATION, GOD, "
                        "CREATURE, EVENT, WEAPON_OR_TOOL, or ORGANIZATION."
                    ),
                },
            },
            "required": ["entity_name"],
        },
    },
}


def entity_info_by_name(
    entity_name: str,
    entity_label: Optional[str] = None,
    database_: Optional[str] = None,
):
    driver = neo4j_driver or ensure_neo4j_driver()
    query = """
    MATCH (e)
    WHERE e.name IS NOT NULL
      AND (
        e.name = toUpper(trim($entity_name))
        OR e.name CONTAINS toUpper(trim($entity_name))
      )
      AND (
        $entity_label IS NULL
        OR any(label IN labels(e) WHERE toUpper(label) = $entity_label)
      )
    WITH
        e,
        CASE
            WHEN e.name = toUpper(trim($entity_name)) THEN 0
            ELSE 1
        END AS exact_rank
    RETURN
        labels(e) AS entity_labels,
        e AS entity,
        [(e)-[r:SUMMARIZED_RELATIONSHIP]->(related) |
            {
                direction: "OUTGOING",
                relationship_type: "SUMMARIZED_RELATIONSHIP",
                summary: r.summary,
                related_entity: related,
                related_entity_labels: labels(related)
            }
        ] +
        [(related)-[r:SUMMARIZED_RELATIONSHIP]->(e) |
            {
                direction: "INCOMING",
                relationship_type: "SUMMARIZED_RELATIONSHIP",
                summary: r.summary,
                related_entity: related,
                related_entity_labels: labels(related)
            }
        ] AS relationships
    ORDER BY exact_rank, e.name
    LIMIT 10
    """

    params = {
        "entity_name": entity_name,
        "entity_label": entity_label.upper() if entity_label else None,
    }

    try:
        records, _, _ = driver.execute_query(query, params, database_=database_)
        return [record.data() for record in records]
    except Exception as e:
        return [f"entity_info_by_name caused an error: {e}"]

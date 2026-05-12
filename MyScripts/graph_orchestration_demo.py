import argparse
import json
from functools import lru_cache

from neo4j.graph import Node, Path, Relationship

from graph_tools import (
    entity_info_by_name_description,
    entity_info_by_name,
    text2cypher_description,
)
from cypher_generation import Text2Cypher, generate_prompt_sections
from runtime import chat, ensure_neo4j_driver, tool_choice


query_update_prompt = """
    You are an expert at updating questions to make them ask for one thing only, more atomic, specific and easier to answer.
    You do this by filling in missing information in the question, using only the extra information provided in previous answers.

    Return exactly one JSON object with this shape and nothing else:
    {
        "question": "question1"
    }

    Rules:
    - Only edit the question if needed.
    - If the original question is already atomic, specific, and easy to answer, keep it unchanged.
    - Do not ask for more information than the original question.
    - Do not return markdown.
"""


def query_update(input: str, answers: list[any]) -> str:
    messages = [
        {"role": "system", "content": query_update_prompt},
        *answers,
        {"role": "user", "content": f"The user question to rewrite: '{input}'"},
    ]
    config = {"response_format": {"type": "json_object"}}
    output = chat(messages, model="gpt-5-nano", config=config)

    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        print("Error decoding JSON in query_update:")
        print(output)
        return input

    question = payload.get("question", input)
    if isinstance(question, str) and question.strip():
        return question.strip()

    print("Unexpected query_update output:")
    print(output)
    return input


def repair_cypher(question: str, bad_cypher: str, error: str, prompt_sections: dict) -> str:
    repair_prompt = f"""
You fix Cypher queries for Neo4j.

Rules:
- Return ONLY the corrected Cypher.
- Do not add explanations.
- Use only the provided schema.

Schema:
{prompt_sections["schema_string"]}

Terminology:
{prompt_sections["terminology_string"]}

Examples:
{prompt_sections["examples_string"]}

Original user question:
{question}

Bad Cypher:
{bad_cypher}

Neo4j error:
{error}
"""
    return chat(
        messages=[{"role": "user", "content": repair_prompt}],
        model="gpt-5-nano",
    ).strip()


@lru_cache(maxsize=8)
def get_prompt_sections(database_: str | None = None) -> dict:
    return generate_prompt_sections(ensure_neo4j_driver(), database_=database_)


def normalize_graph_value(value):
    if isinstance(value, Node):
        return {
            "kind": "node",
            "labels": list(value.labels),
            "properties": dict(value),
        }
    if isinstance(value, Relationship):
        return {
            "kind": "relationship",
            "type": value.type,
            "properties": dict(value),
        }
    if isinstance(value, Path):
        return {
            "kind": "path",
            "nodes": [{"labels": list(n.labels), "properties": dict(n)} for n in value.nodes],
            "relationships": [{"type": r.type, "properties": dict(r)} for r in value.relationships],
        }
    if isinstance(value, list):
        return [normalize_graph_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_graph_value(v) for k, v in value.items()}
    return value


def normalize_rows(rows):
    return [{k: normalize_graph_value(v) for k, v in row.items()} for row in rows]


def orchestrated_text2cypher(
    question: str,
    prompt_sections: dict,
    max_retries: int = 2,
    database_: str | None = None,
):
    driver = ensure_neo4j_driver()
    t2c = Text2Cypher(driver=driver, schema_mode="core_schema", database_=database_)
    t2c.set_prompt_section("question", question)
    t2c.set_prompt_section("terminology", prompt_sections["terminology_string"])
    t2c.set_prompt_section("examples", prompt_sections["examples_string"])

    cypher = t2c.generate_cypher()
    last_error = None

    for _ in range(max_retries + 1):
        try:
            driver.execute_query(f"EXPLAIN {cypher}", database_=database_)
            records, _, _ = driver.execute_query(cypher, database_=database_)
            rows = normalize_rows([r.data() for r in records[:10]])
            return {
                "tool": "text2cypher",
                "question": question,
                "cypher": cypher,
                "rows": rows,
                "error": None,
            }
        except Exception as e:
            last_error = str(e)
            cypher = repair_cypher(question, cypher, last_error, prompt_sections)

    return {
        "tool": "text2cypher",
        "question": question,
        "cypher": cypher,
        "rows": [],
        "error": last_error,
    }


tool_picker_prompt = """
    You must choose the single best next tool for answering the user's question.

    Routing rules:
    - Use entity_info_by_name only when the question is mainly about a specific named entity.
    - Use text2cypher for:
      - relationships between entities
      - counts, lists, rankings, filtering, aggregation
      - questions that cannot be answered from one entity lookup alone
      - cases where prior entity lookup was incomplete
    - Use answer_given only if the previous tool results already fully answer the original user question.

    Important:
    - Prefer entity_info_by_name first for single-entity lookup.
    - Prefer text2cypher for multi-hop graph reasoning.
    - Do not call answer_given unless the answer is already explicitly present in prior results.
    - Make exactly one tool call.
"""


def build_tools(prompt_sections: dict, database_: str | None = None):
    return {
        "entity_info_by_name": {
            "description": entity_info_by_name_description,
            "function": lambda entity_name, entity_label=None: entity_info_by_name(
                entity_name,
                entity_label,
                database_=database_,
            ),
        },
        "text2cypher": {
            "description": text2cypher_description,
            "function": lambda question: orchestrated_text2cypher(
                question,
                prompt_sections,
                database_=database_,
            ),
        },
    }


def handle_tool_calls(tools: dict[str, any], llm_tool_calls: list[dict[str, any]]):
    output = []
    if llm_tool_calls:
        for tool_call in llm_tool_calls:
            tool_name = tool_call.function.name
            function_to_call = tools[tool_name]["function"]
            function_args = json.loads(tool_call.function.arguments)

            try:
                res = function_to_call(**function_args)
                output.append(
                    {
                        "tool": tool_name,
                        "args": function_args,
                        "result": res,
                    }
                )
            except Exception as e:
                output.append(
                    {
                        "tool": tool_name,
                        "args": function_args,
                        "result": None,
                        "error": str(e),
                    }
                )
    return output


MAX_TOOL_ITEMS = 5
MAX_TOOL_CHARS = 12000


def compact_tool_output(response, max_items: int = MAX_TOOL_ITEMS, max_chars: int = MAX_TOOL_CHARS):
    text = json.dumps(response, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[TRUNCATED FOR CONTEXT BUDGET]..."
    return text


def route_question(question: str, tools: dict[str, any], answers: list[dict[str, str]]):
    router_output = tool_choice(
        [
            {"role": "system", "content": tool_picker_prompt},
            *answers,
            {"role": "user", "content": f"The user question to find a tool to answer: '{question}'"},
        ],
        model="gpt-5-nano",
        tools=[tool["description"] for tool in tools.values()],
    )

    if not router_output["tool_calls"]:
        return [
            {
                "tool": "router_failure",
                "args": {},
                "result": None,
                "error": f"no_tool_call; finish_reason={router_output['finish_reason']}; content={router_output['content']!r}",
            }
        ]

    return handle_tool_calls(tools, router_output["tool_calls"])


def handle_user_input(input: str, tools: dict[str, any], answers=None, mode: str = "agentic"):
    if answers is None:
        answers = []

    updated_question = query_update(input, answers) if mode == "agentic" else input
    response = route_question(updated_question, tools, answers)
    compact_response = compact_tool_output(
        {
            "question": updated_question,
            "tool_results": response,
        }
    )

    answers.append({"role": "assistant", "content": compact_response})
    return answers


answer_critique_prompt = """
    You are an expert at identifying whether a question has been fully answered or whether more information is needed.

    The user will provide:
    - the original question
    - previously gathered answer information

    Your job:
    - If the existing information is sufficient, return exactly:
      {"questions": []}
    - If something is missing, return exactly:
      {"questions": ["question1", "question2"]}

    Rules:
    - Every follow-up question must be complete, atomic, and specific.
    - Return exactly one JSON object with a top-level "questions" field.
    - Do not return markdown.
    - Do not return any key other than "questions".
"""


def critique_answers(question: str, answers: list[dict[str, str]]) -> list[str]:
    messages = [
        {"role": "system", "content": answer_critique_prompt},
        *answers,
        {"role": "user", "content": f"The original user question to answer: {question}"},
    ]
    config = {"response_format": {"type": "json_object"}}
    output = chat(messages, model="gpt-5-nano", config=config)

    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        print("Error decoding JSON in critique_answers:")
        print(output)
        return []

    questions = payload.get("questions", [])
    if isinstance(questions, list):
        return [q.strip() for q in questions if isinstance(q, str) and q.strip()]
    if isinstance(questions, str) and questions.strip():
        return [questions.strip()]

    print("Unexpected critique_answers output:")
    print(output)
    return []


main_prompt = """
Your job is to answer the user's question using only the provided tool results.
If the available information is insufficient, say exactly what is still missing.
Do not invent facts.
"""


def run_stable_orchestration(input: str, tools: dict[str, any]) -> str:
    answers = handle_user_input(input, tools, answers=[], mode="stable")
    return chat(
        [
            {"role": "system", "content": main_prompt},
            *answers,
            {"role": "user", "content": f"The user question to answer: {input}"},
        ],
        model="gpt-5-nano",
    )


def run_agentic_orchestration(input: str, tools: dict[str, any], max_steps: int = 4) -> str:
    answers: list[dict[str, str]] = []
    pending_questions = [input]

    for _ in range(max_steps):
        if not pending_questions:
            break

        next_q = pending_questions.pop(0)
        answers = handle_user_input(next_q, tools, answers, mode="agentic")

        critique = critique_answers(input, answers)
        if not critique:
            break

        seen = set(pending_questions)
        pending_questions.extend([q for q in critique if q not in seen])

    return chat(
        [
            {"role": "system", "content": main_prompt},
            *answers,
            {"role": "user", "content": f"The user question to answer: {input}"},
        ],
        model="gpt-5-nano",
    )


def main(input: str, max_steps: int = 4, database_: str | None = None, mode: str = "agentic"):
    prompt_sections = get_prompt_sections(database_)
    tools = build_tools(prompt_sections, database_=database_)

    if mode == "stable":
        return run_stable_orchestration(input, tools)
    return run_agentic_orchestration(input, tools, max_steps=max_steps)


def cli():
    parser = argparse.ArgumentParser(description="Orchestration demo")
    parser.add_argument(
        "--question",
        default="How is ULYSSES related to NEPTUNE?",
        help="Question to answer",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Optional Neo4j database name to query",
    )
    parser.add_argument(
        "--mode",
        default="agentic",
        choices=["agentic", "stable"],
        help="Orchestration mode",
    )
    args = parser.parse_args()
    print(main(args.question, database_=args.database, mode=args.mode))


if __name__ == "__main__":
    cli()

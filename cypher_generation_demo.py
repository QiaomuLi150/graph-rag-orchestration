import argparse

from cypher_generation import Text2Cypher, generate_prompt_sections
from runtime import ensure_neo4j_driver


def main():
    parser = argparse.ArgumentParser(description="Text-to-Cypher demo")
    parser.add_argument(
        "--question",
        default="How is ULYSSES related to NEPTUNE?",
        help="Question to convert into Cypher",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Optional Neo4j database name to query",
    )
    args = parser.parse_args()

    driver = ensure_neo4j_driver()
    sections = generate_prompt_sections(driver, database_=args.database)

    t2c = Text2Cypher(driver=driver, schema_mode="core_schema", database_=args.database)
    t2c.set_prompt_section("question", args.question)
    t2c.set_prompt_section("terminology", sections["terminology_string"])
    t2c.set_prompt_section("examples", sections["examples_string"])

    cypher = t2c.generate_cypher()
    print(cypher)


if __name__ == "__main__":
    main()

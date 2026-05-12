# KG-RAG Demo Scripts

This folder contains standalone demo scripts rewritten from the notebook workflow.

## Demos

- `entity_extraction_demo.py` - extract entities and relationships from a small sample text and import them into Neo4j.
- `cypher_generation_demo.py` - generate Cypher from a natural-language question using schema-aware prompt sections.
- `graph_orchestration_demo.py` - route a question through entity lookup, Cypher generation, repair, and answer synthesis.

## Dependencies

Install the Python dependencies listed in `requirements.txt`.

## Environment

Set these variables before running the demos:

- `OPENAI_API_KEY`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

## Notes

- The demos expect a Neo4j database with APOC installed.
- The extraction demo also expects the graph schema used by the notebooks, including `__Entity__`, `__Chunk__`, `RELATIONSHIP`, and the summary/community structures.
- The orchestration and Text-to-Cypher demos generate prompts from the live Neo4j schema.

## Example commands

```bash
python entity_extraction_demo.py
python cypher_generation_demo.py --question "How is MERCURY related to MINERVA?"
python graph_orchestration_demo.py --question "Who is MERCURY?"
```

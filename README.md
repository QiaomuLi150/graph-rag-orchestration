# Graph RAG Orchestration

This repository contains a compact tutorial bundle for a graph-based RAG system built on top of Neo4j and OpenAI models. It is organized as three demos plus one integrated notebook that walks through the full workflow end to end.

## What is included

- `entity_extraction_demo.py`: entity and relationship extraction from source text
- `cypher_generation_demo.py`: schema-aware Text-to-Cypher generation
- `graph_orchestration_demo.py`: orchestration logic that routes questions to the best available method
- `integrated_graph_rag_demo.ipynb`: the full tutorial notebook that connects the complete pipeline
- `extraction_tools.py`, `cypher_generation.py`, `graph_tools.py`, `neo4j_schema.py`, `runtime.py`: supporting modules

## System Overview

The tutorial shows a complete graph RAG workflow:

1. Download or load source text.
2. Clean and chunk the text.
3. Extract entities and relationships.
4. Store the graph in a dedicated Neo4j database.
5. Summarize entities, relationships, and communities.
6. Build embeddings and vector indexes.
7. Answer questions through multiple graph-based methods.
8. Compare orchestration modes and question-answering strategies.

## Orchestration Modes

The orchestration layer provides two modes:

- `agentic`: keeps the original multi-step loop with question rewriting, routing, critique, and answer synthesis.
- `stable`: uses the same core routing path but skips the extra rewrite/critique loop for a lighter and more predictable run.

Both modes share the same base methods and the same Neo4j-backed data, so the difference is in control flow rather than in graph functionality.

## Question Answering Methods

The tutorial compares these methods:

- `entity_info_by_name` for a single named entity
- `local_search` for local evidence in chunks and summaries
- `global_retriever` for higher-level community synthesis
- `Text2Cypher` for structured graph queries
- `orchestration_main` for automatic routing across the available methods

## Requirements

Install the Python dependencies listed in `requirements.txt`.

You also need:

- Neo4j running locally or remotely
- OpenAI API access
- the Neo4j credentials configured through environment variables

## Configuration

The demos expect these environment variables:

- `OPENAI_API_KEY`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

## How to run

Typical workflow:

```bash
python entity_extraction_demo.py
python cypher_generation_demo.py
python graph_orchestration_demo.py
```

For the notebook tutorial, open `integrated_graph_rag_demo.ipynb` in Jupyter and run the cells top to bottom.

## Repo Layout

- `integrated_graph_rag_demo.ipynb`: full tutorial notebook
- `entity_extraction_demo.py`: extraction demo
- `cypher_generation_demo.py`: Cypher demo
- `graph_orchestration_demo.py`: orchestration demo
- `sample_data/`: small bundled example text

## Notes

The repository is designed as a public tutorial bundle. The notebook and scripts are intentionally structured so readers can follow the full pipeline from source text to graph construction to question answering.

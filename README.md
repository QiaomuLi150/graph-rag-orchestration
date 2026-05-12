# Graph RAG Orchestration

This repository is a compact tutorial bundle for a graph-based RAG system built on Neo4j and OpenAI models.
It presents the full pipeline from source text to graph construction to question answering, with a notebook that
ties the pieces together and three scripts that demonstrate the main stages.

## Overview

![Graph RAG pipeline](assets/pipeline.svg)

The system follows a simple flow:

1. Load or download source text.
2. Clean and chunk the text.
3. Extract entities and relationships.
4. Store the graph in a dedicated Neo4j database.
5. Build summaries, embeddings, and retrieval indexes.
6. Answer questions through graph-aware methods.
7. Route questions with orchestration.

## What’s Included

- `entity_extraction_demo.py`: entity and relationship extraction from source text
- `cypher_generation_demo.py`: schema-aware Text-to-Cypher generation
- `graph_orchestration_demo.py`: orchestration logic that selects the best question-answering method
- `integrated_graph_rag_demo.ipynb`: the end-to-end tutorial notebook
- `extraction_tools.py`, `cypher_generation.py`, `graph_tools.py`, `neo4j_schema.py`, `runtime.py`: supporting modules

## Orchestration Modes

![Orchestration modes](assets/orchestration_modes.svg)

The orchestration layer provides two modes:

- `agentic`: the original multi-step loop with question rewriting, routing, critique, and synthesis.
- `stable`: the lighter mode that keeps the same routing path but skips the extra rewrite/critique loop.

Both modes use the same graph-backed data and the same base methods. The difference is control flow.

## Question Answering Methods

The tutorial compares these methods:

- `entity_info_by_name` for a single named entity
- `local_search` for local evidence in chunks and summaries
- `global_retriever` for higher-level community synthesis
- `Text2Cypher` for structured graph queries
- `orchestration_main` for automatic routing across the available methods

## Requirements

Install the Python dependencies in `requirements.txt`.

You also need:

- Neo4j running locally or remotely
- OpenAI API access
- the Neo4j credentials configured through environment variables

## Configuration

Set these environment variables before running the demos:

- `OPENAI_API_KEY`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

## How to Run

Run the individual demos directly:

```bash
python entity_extraction_demo.py
python cypher_generation_demo.py
python graph_orchestration_demo.py
```

Open `integrated_graph_rag_demo.ipynb` in Jupyter to follow the complete tutorial step by step.

## Repository Layout

- `integrated_graph_rag_demo.ipynb`: full tutorial notebook
- `entity_extraction_demo.py`: extraction demo
- `cypher_generation_demo.py`: Cypher demo
- `graph_orchestration_demo.py`: orchestration demo
- `sample_data/`: bundled example text
- `assets/`: diagrams used in the README

## Notes

This repository is intended as a public tutorial bundle. The code is structured so readers can follow the full
workflow from source text to graph ingestion to question answering without jumping between unrelated notebooks.

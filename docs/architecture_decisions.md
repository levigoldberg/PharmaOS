# Architecture Decisions

## Orchestration

Use OpenAI Agents SDK for agent orchestration.

The system should use a control tower plus specialist agents. Workflows should be semi-deterministic: code controls the lifecycle, agents perform bounded reasoning, and tools perform deterministic retrieval/calculation.

Do not use LangChain or LangGraph unless the current orchestrator becomes too hard to maintain.

## Scientific Memory

Use SQLite first.

Scientific Memory stores:

- runs
- sources
- entities
- evidence claims
- agent outputs
- validation results
- confidence flags
- human gates
- final reports

Do not use a vector database until document retrieval requires it.

Do not use Neo4j or Postgres unless SQLite becomes a real limitation.

## Agents vs tools

Agents:

- reason
- synthesize
- prioritize
- draft
- critique

Tools:

- call APIs
- parse responses
- normalize data
- calculate values
- retrieve documents
- validate outputs

Agents should not invent facts that should come from tools.

## Typed objects

Agents and workflows should communicate through Pydantic models, not long unstructured paragraphs.

## Validation

Every workflow output should pass through:

1. schema validation
2. source coverage check
3. numeric provenance check
4. cross-agent consistency check where relevant
5. confidence scoring
6. human-gate assignment
7. Scientific Memory write

Do not hide failed validators.

## Interfaces

Primary interface is CLI.

Optional UI or dashboard can be added after the CLI workflows and report outputs are working.

## Data policy

Use real APIs and open-source tools wherever possible.

If a workflow cannot run without unavailable regulated or physical-system data, return a typed not_implemented result and document the missing connector.

## Reuse strategy for existing `levigoldberg/ClinicalTrialIntel` repo

The existing `levigoldberg/ClinicalTrialIntel` repository should be treated as a source of reusable implementations for one workflow: Trial Intelligence / Due Diligence.

Do not copy its full architecture directly.

Reuse or adapt:

- deterministic API clients
- Pydantic schemas where compatible
- asset identity logic
- PoS lookup pattern
- pricing/commercial/rNPV calculators
- source metadata and confidence patterns
- tests where portable

Refactor as needed so reused code fits the new architecture:

- tools live under `src/pharma_os/tools/`
- agents live under `src/pharma_os/agents/`
- workflows live under `src/pharma_os/workflows/`
- outputs are saved to Scientific Memory
- validation results and human gates are first-class objects
- workflows should not be globally NCT-centered even if one workflow accepts NCT IDs

# AGENTS.md

## Mission

Build a lean AI-native pharma operating system prototype.

The system coordinates specialized agents across the pharma lifecycle: discovery, tox/PKPD, trial intelligence, due diligence, protocol writing, enrollment, trial execution, BioFactory/manufacturing, launch/PV, and regulatory/audit.

The goal is multiple real-data vertical slices sharing one architecture.

## Core lifecycle

Every workflow should follow:

input
→ deterministic tools/API retrieval
→ specialist agent reasoning
→ typed Pydantic output
→ validation/confidence scoring
→ Scientific Memory write
→ report/audit output

Automate by default. Add human gates only for high-risk scientific, clinical, regulatory, safety, consent, GMP, promotional, or major capital-allocation decisions.

## Framework decisions

- Use OpenAI Agents SDK for agents and orchestration.
- Use Pydantic v2 for typed schemas.
- Use SQLite for Scientific Memory.
- Use httpx for API clients.
- Use pytest for tests.
- Use CLI as the primary interface.
- Streamlit or another UI may be added only after CLI workflows work.
- Do not add LangChain, LangGraph, Neo4j, FastAPI, React, Docker, Postgres, or a vector database unless explicitly justified.

## Repo layout

- `docs/`: project brief, architecture decisions, source prompt, build plan.
- `src/pharma_os/cli.py`: command-line entry point.
- `src/pharma_os/orchestrator.py`: control tower orchestration.
- `src/pharma_os/schemas.py`: Pydantic models.
- `src/pharma_os/memory.py`: SQLite Scientific Memory.
- `src/pharma_os/validators.py`: validation and confidence scoring.
- `src/pharma_os/report.py`: final reports.
- `src/pharma_os/agents/`: specialist agents.
- `src/pharma_os/tools/`: deterministic API clients and calculators.
- `src/pharma_os/workflows/`: vertical slices.
- `tests/`: unit and workflow tests.

## Priority workflows

Implement multiple workflows where real data is available:

1. Discovery / target landscape
2. Tox / PKPD / safety screen
3. Trial intelligence + due diligence
4. Protocol + enrollment feasibility
5. Launch / RWE / pharmacovigilance
6. BioFactory only if real public tools/data make it feasible

## Existing reusable repo

Before implementing the Trial Intelligence / Due Diligence workflow, inspect:

- `panoptic-trial-intel`

Use it as reusable source material for that workflow only. Do not copy its whole NCT-centered architecture into this project.

Likely reusable pieces:

- ClinicalTrials.gov client
- RxNorm normalization
- PubMed / Europe PMC retrieval
- asset identity logic
- patent / LOE workflow
- PoS classification + deterministic lookup
- pricing, commercial model, and rNPV calculators
- report aggregation and test patterns

Adapt reused code to this project’s architecture:

- deterministic retrieval/calculation belongs in `tools/`
- bounded reasoning belongs in `agents/`
- workflow sequencing belongs in `workflows/`
- outputs, evidence claims, validation results, confidence flags, and human gates must be saved to Scientific Memory

## API/tool rules

API tools must:

- use real APIs where possible
- use httpx with timeouts
- return typed objects
- preserve source metadata
- handle errors explicitly
- distinguish missing data from failed calls
- avoid LLM calls inside API clients
- avoid silent fallback data

Priority APIs:

- ClinicalTrials.gov
- PubMed E-utilities
- openFDA labels
- openFDA FAERS
- PubChem
- DailyMed
- ChEMBL
- RxNorm
- FDA / EMA / ICH regulatory documents where feasible

## Agent rules

Each agent must:

- have a narrow role
- use tools instead of inventing facts
- return typed Pydantic output
- include evidence claims
- include confidence flags
- mark required human gates
- avoid unsupported numeric claims
- avoid final clinical, regulatory, GMP, promotional, or investment decisions

Agents should communicate through typed objects whenever possible.

## Scientific Memory

Scientific Memory stores:

- runs
- sources
- entities
- evidence claims
- agent outputs
- validation results
- human gates
- final reports

Do not store unsupported free text as fact. Store claims with source metadata, confidence, and validation status.

## Validation rules

After each workflow or agent output:

- validate schema
- check source coverage
- check numeric provenance
- check cross-agent consistency where relevant
- assign confidence flags
- assign human gates
- save validation results

Failed validation should be visible in reports.

## Data policy

Prefer real API data and open-source tools.

Do not create fake patient, site, batch, EHR, CTMS, EDC, MES, LIMS, or manufacturing data by default. If a module requires unavailable regulated data or physical systems, return a typed `not_implemented` result with the missing connector or required system.

## Commands

Use these commands when available:

```bash
python -m pytest
python -m pharma_os run <workflow> [args]
python -m pharma_os report --run-id <RUN_ID>
```

# AGENTS.md

## Mission

Build a lean AI-native pharma operating system prototype.

PharmaOS coordinates specialized agents and deterministic tools across the pharma lifecycle. The current executable clinical-development slice is Agent 3 clinical outcome prediction, Agent 4 due diligence, and Agent 5 protocol design, governed by Scientific Memory and the Control Tower.

## Current Architecture

The current orchestration architecture is:

Objective -> Scientific Memory -> Control Tower -> Capability Registry -> run/reuse/refresh/skip/block -> specialist workflow -> validation/human gates -> memory update -> replan.

The Control Tower must choose the minimum justified path for the objective. It should reason over the typed Scientific State Snapshot: pending downstream decision, evidence requirements, requirement satisfaction, unresolved or contradictory claims, critical gaps, stale or incompatible artifacts, human gates, and blocked capabilities. It should reuse compatible Scientific Memory artifacts, refresh stale or invalidated artifacts, skip explicitly skipped capabilities, block unavailable modules with missing connectors, and replan after material state changes.

Control Tower prompting is not the source of truth for evidence logic. Deterministic code builds Scientific State, assesses freshness and compatibility, detects evidence gaps and gates, validates plans, and blocks skeleton modules. The live Control Tower uses the OpenAI Agents SDK for planning/replanning over that state. The deterministic fallback planner exists for offline operation and tests and should stay minimal.

AI execution mode must be impossible to miss. Important reasoning traces, agent-output envelopes, workflow outputs, Control Tower records, and reports must surface one of: `live_agent`, `direct_llm`, `deterministic_fallback`, or `reused_artifact`. Do not rely only on trace metadata for fallback visibility.

Implemented executable workflows:

- `clinical_outcome_prediction`: Agent 3 clinical outcome/risk context for one NCT ID.
- `due_diligence`: Agent 4 clinical-stage due diligence using Agent 3 handoff plus public/source-backed diligence tools.
- `protocol_design`: Agent 5 draft next-study protocol design brief using Agent 3/4 handoffs and analog benchmarking.

Compatibility route:

- `trial_intelligence`: legacy CLI route over the internal Agent 3 trial-landscape component. It is not a separate top-level LLM agent.

Registered non-executable skeletons:

- `discovery`
- `tox_pkpd_safety`
- `enrollment_feasibility`
- `trial_execution`
- `manufacturing_biofactory`
- `launch_pv`
- `regulatory_quality_audit`

Skeleton capabilities are registry entries with evidence requirements. They allow the Scientific State and Control Tower to explain unavailable modules and missing connectors; they must not be executed or filled with fake data.

## Core Lifecycle

Executable workflows should follow:

input
-> deterministic tools/API retrieval
-> specialist agent reasoning
-> typed Pydantic output
-> validation/confidence scoring
-> Scientific Memory write
-> report/audit output

Control Tower orchestration wraps those workflows with planning, artifact reuse, refresh decisions, blocking, and replanning.

Automate by default. Add human gates for high-risk scientific, clinical, regulatory, safety, consent, GMP, promotional, or major capital-allocation decisions.

## Documentation Currency Rule

Whenever a change affects architecture, workflows, capabilities, runtime behavior, CLI commands, implementation status, data contracts, validation behavior, or agent/tool boundaries, update the relevant documentation in the same change.

`AGENTS.md`, `README.md`, and architecture docs must not knowingly remain stale after implementation changes. If implementation and docs disagree, fix the docs or explicitly document the temporary gap before finishing the change.

## Framework Decisions

- Use OpenAI Agents SDK for bounded agent reasoning and Control Tower planning.
- Use direct structured OpenAI calls only for narrow one-shot reasoning where agentic coordination is unnecessary.
- Preserve deterministic fallbacks when configured or when live execution fails, and surface fallback counts in typed outputs and reports.
- Keep Python workflow functions as the deterministic control plane.
- Use Pydantic v2 for typed schemas.
- Use SQLite for Scientific Memory.
- Use httpx for API clients.
- Use pytest for tests.
- Use CLI as the primary interface.
- Streamlit or another UI may be added only after CLI workflows work.
- Do not add LangChain, LangGraph, Neo4j, FastAPI, React, Docker, Postgres, or a vector database unless explicitly justified.

## Repo Layout

- `docs/`: project brief, architecture decisions, and historical lean build plan.
- `src/pharma_os/cli.py`: command-line entry point.
- `src/pharma_os/orchestrator.py`: direct workflow runs plus Control Tower orchestration.
- `src/pharma_os/control_tower.py`: Control Tower planning primitives and plan validation.
- `src/pharma_os/control_tower_state.py`: deterministic Scientific State decision, requirement, gap, and blocking helpers.
- `src/pharma_os/registry.py`: Capability Registry for implemented workflows and skeleton modules.
- `src/pharma_os/schemas.py`: Pydantic models.
- `src/pharma_os/memory.py`: SQLite Scientific Memory.
- `src/pharma_os/validators.py`: validation and confidence scoring.
- `src/pharma_os/report.py` and `src/pharma_os/html_report.py`: reports and run viewers.
- `src/pharma_os/agents/`: specialist agents and SDK-backed bounded reasoning.
- `src/pharma_os/tools/`: deterministic API clients, adapters, and calculators.
- `src/pharma_os/components/`: reusable deterministic workflow components.
- `src/pharma_os/workflows/`: executable vertical slices.
- `tests/`: unit and workflow tests.

## Existing Reusable Repo

The old `levigoldberg/ClinicalTrialIntel` project is reusable source material for clinical trial intelligence and due diligence only. Do not copy its whole NCT-centered architecture into PharmaOS.

Reusable pieces include ClinicalTrials.gov, RxNorm, PubMed/Europe PMC, asset identity, patent/LOE, PoS, pricing, commercial model, rNPV, report aggregation, and test patterns.

Adapt reused code to this architecture:

- deterministic retrieval/calculation belongs in `tools/`
- bounded reasoning belongs in `agents/`
- workflow sequencing belongs in `workflows/`
- shared reusable logic belongs in `components/`
- outputs, evidence claims, validation results, confidence flags, and human gates must be saved to Scientific Memory

## API/Tool Rules

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

## Agent Rules

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

## Validation Rules

After each workflow or agent output:

- validate schema
- check source coverage
- check numeric provenance
- check cross-agent consistency where relevant
- assign confidence flags
- assign human gates
- save validation results

Failed validation should be visible in reports.

## Data Policy

Prefer real API data and open-source tools.

Do not create fake patient, site, batch, EHR, CTMS, EDC, MES, LIMS, or manufacturing data by default. If a module requires unavailable regulated data or physical systems, return a typed `not_implemented` result or a blocked skeleton capability with the missing connector or required system.

## Shared Artifact Reuse Rule

Before any workflow calls an external API or recomputes a derived object, it must first check Scientific Memory or the canonical shared component for an existing valid artifact with the same stable key.

Stable keys should include the artifact type plus relevant identifiers such as NCT ID, asset name, indication, phase, source name, and query parameters.

Reuse existing valid artifacts unless a refresh flag is explicitly supplied or the artifact is missing, schema-invalid, source-stale, or incompatible with the requested inputs.

When a workflow reuses an artifact, persist a typed reference to the original run ID, output ID, artifact ID, and source IDs. Do not copy unstructured text as the handoff.

When a workflow refreshes an artifact, record why it was refreshed.

Downstream agents must consume typed artifacts from upstream workflows or shared components instead of repeating upstream retrieval, reasoning, or calculations.

## Commands

Use these commands when available:

```bash
python -m pytest
python -m pharma_os run <workflow> [args]
python -m pharma_os orchestrate --goal "<objective>" [args]
python -m pharma_os report --run-id <RUN_ID>
```

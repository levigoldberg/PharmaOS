# Architecture Decisions

## Orchestration

Use OpenAI Agents SDK for agent orchestration.

The system should use a control tower plus specialist agents. Workflows should be semi-deterministic: code controls the lifecycle, agents perform bounded reasoning, and tools perform deterministic retrieval/calculation.

Do not use LangChain or LangGraph unless the current orchestrator becomes too hard to maintain.

The current Control Tower flow is:

Objective -> Scientific Memory -> Control Tower -> Capability Registry -> run/reuse/refresh/skip/block -> specialist workflow -> validation/human gates -> memory update -> replan.

The Control Tower chooses the minimum justified path. It reuses compatible artifacts from Scientific Memory, refreshes stale or invalidated artifacts, blocks registered skeleton capabilities with missing connector reasons, and replans after material state changes.

The Control Tower plans over a typed Scientific State Snapshot rather than keyword routing alone. Deterministic code infers the pending downstream decision, selects registry evidence requirements, assesses which artifacts satisfy each requirement, marks stale or incompatible artifacts, surfaces unresolved or contradictory claims, records critical evidence gaps and human gates, and lists blocked capabilities. The live Control Tower uses the OpenAI Agents SDK to plan and replan over this state. The offline deterministic fallback planner remains intentionally minimal for tests and offline operation.

Execution-plan validation is deterministic. It checks dependency ordering, missing dependencies, skeleton-module execution attempts, reuse of missing or incompatible artifacts, unjustified refresh/rerun requests, blocking human gates, unrelated run/refresh steps that do not address pending decision requirements, and reuse steps that cite requirements their artifact does not satisfy.

The current Capability Registry includes three implemented executable workflows: Agent 3 `clinical_outcome_prediction`, Agent 4 `due_diligence`, and Agent 5 `protocol_design`. It also includes registered non-executable skeletons for discovery, tox/PK-PD, enrollment feasibility, trial execution, manufacturing/biofactory, launch/PV, and regulatory/quality/audit.

AI execution mode is a typed audit field, not hidden metadata. The allowed modes are `live_agent` for Agents SDK coordination, `direct_llm` for narrow direct structured OpenAI calls, `deterministic_fallback` for offline/configured or failure fallbacks, and `reused_artifact` for compatible Scientific Memory reuse. Reports and human-readable audit views must summarize execution counts.

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

## Agent 3 and Agent 4 boundary

`clinical_outcome_prediction` is the canonical Agent 3 workflow. It owns clinical outcome prediction, including deterministic ClinicalTrials.gov trial identity, design features, endpoint/enrollment risk, historical PoS, safety context, and an internal trial-landscape component for comparator context.

The legacy `trial_intelligence` CLI route is compatibility-only Agent 3 landscape mode. It calls `src/pharma_os/components/trial_landscape.py` and must not evolve as a separate LLM-driven top-level agent.

`due_diligence` remains separate as Agent 4. It owns patents, pricing, commercial model, market sizing, BD logic, and rNPV. Agent 3 schemas and reports must not add Agent 4 fields such as `rnpv`, `commercial_model`, `patent_exclusivity`, patents, pricing, or market sizing.

Agent 4 consumes Agent 3 through a typed handoff, not by recomputing Agent 3-only clinical-risk logic. `due_diligence` reuses the latest completed, non-failed `clinical_outcome_prediction` output for the same NCT ID unless `refresh_agent3` is requested; otherwise it generates and persists a new Agent 3 run. The Agent 4 output stores `agent3_handoff` and `clinical_risk_summary`, then runs reusable cross-agent consistency validation.

The Clinical Stage Due Diligence Agent expands Agent 4 with deterministic sections only from allowed sources: Agent 3 output, ClinicalTrials.gov, PubMed, openFDA labels, Lens, local PoS/WAC workbooks, and reviewed/config assumptions. `clinical_evidence`, `competitive_landscape`, `safety_label_summary`, `patent_loe_review`, `red_flags`, and `asset_memo` are assembled inside PharmaOS and persisted to Scientific Memory. The asset memo is a draft requiring human review and must not contain final recommendations, approvals, legal/regulatory conclusions, or invented values.

## Agent 5 Protocol Design Brief boundary

`protocol_design` is the canonical Agent 5 workflow. It consumes typed Agent 3 `clinical_outcome_prediction` and Agent 4 `due_diligence` outputs, then produces a source-grounded `ProtocolDesignBrief`. It is not a full protocol, IRB-ready protocol, submission-ready protocol, or approval artifact.

Agent 5 is built around analog trial benchmarking. Bounded helper subagents create a CT.gov search plan, select analog trials, draft strategy/eligibility/schedule sections, and review gaps. Deterministic tools execute CT.gov retrieval, deduplicate and normalize candidates, calculate benchmark statistics, assemble template sections, validate source/numeric support, persist Scientific Memory artifacts, and create the human gate.

Agent 5 analog benchmarking must prefer structured ClinicalTrials.gov design and arm fields, including allocation, intervention model, masking, observational model, number of arms, arm groups, and intervention-to-arm mappings. Free-text trial-title, intervention-name, and eligibility heuristics are fallback logic only when structured CT.gov fields are missing or insufficient.

Allowed Agent 5 data sources are Agent 3 output, Agent 4 output, ClinicalTrials.gov, PubMed abstracts/metadata, openFDA label context, and local protocol templates/config checklists. Agent 5 must not add EHR, claims, OMOP, FHIR, patient matching, AACT, Trialtrove, GlobalData, SEC, EMA EPARs, Orange Book, DrugBank, proprietary data, fake patient/site/enrollment data, or final approval logic. Missing values become flags. The `AnalogBenchmarkBundle` is a first-class field in `ProtocolDesignOutput` and is persisted in Scientific Memory for later agents.

## Due-Diligence Data Layers

Agent 4 due diligence is organized only around real data layers currently implemented in this repo:

- Clinical trial data: `tools/clinicaltrials.py` remains the single ClinicalTrials.gov retrieval layer. `tools/asset_identity.py` resolves trial asset, sponsor, indication, and modality from normalized CT.gov records, RxNorm, and shared rules.
- Discovery / IP-adjacent data: `tools/patents_lens.py` contains the Lens-only patent/LOE workflow and reviewed LOE fallback flags.
- Commercial and safety data: `tools/pos.py`, `tools/pricing.py`, `tools/commercial_model.py`, and `tools/rnpv.py` contain the local PoS workbook lookup, local WAC/openFDA pricing evidence, deterministic commercial model, and deterministic rNPV.

`tools/due_diligence.py` is a compatibility facade, not the implementation home. Future capabilities may be registered as non-executable skeletons in the Capability Registry so the Control Tower can block them explicitly, but do not create executable tools, fake connectors, or empty implementation packages for Discovery, Safety/Translational, RWD, Manufacturing, broader Commercial/Safety, proprietary databases, or other future sources until real data/tools exist.

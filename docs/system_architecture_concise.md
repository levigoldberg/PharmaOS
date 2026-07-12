# System Architecture Description

A user objective or ClinicalTrials.gov NCT ID runs through shared orchestration, memory, capability selection, specialist workflows, validation, human gates, and report generation. PharmaOS currently executes Agent 3 clinical outcome prediction, Agent 4 due diligence, and Agent 5 protocol design; other lifecycle modules are registered skeletons that block rather than fabricate outputs.

## Control Tower and Scientific Memory

The Control Tower flow is `Objective -> Scientific Memory -> Control Tower -> Capability Registry -> run/reuse/refresh/skip/block -> specialist workflow -> validation/human gates -> memory update -> replan`.

Scientific Memory is SQLite. It stores runs, sources, claims, agent outputs, agent traces, validation results, confidence flags, human gates, reports, and workflow output JSON. Before running a workflow, the Control Tower asks memory for compatible artifacts with the same request context. Fresh compatible artifacts can be reused; stale, superseded, incompatible, or force-refreshed artifacts can be refreshed.

## Capability Registry

The registry is the source of truth for what can run. `clinical_outcome_prediction`, `due_diligence`, and `protocol_design` are implemented executable workflows. `due_diligence` depends on Agent 3, and `protocol_design` depends on Agent 3 and Agent 4.

`discovery`, `tox_pkpd_safety`, `enrollment_feasibility`, `trial_execution`, `manufacturing_biofactory`, `launch_pv`, and `regulatory_quality_audit` are non-executable skeleton capabilities. They exist so the Control Tower can explain missing connectors and block unsafe requests instead of inventing regulated or unavailable data.

## Request Understanding

Natural-language orchestration uses `RequestUnderstandingAgent` to extract the target capability, decision type, NCT ID, asset name, indication, reviewed assumptions, refresh hints, skip hints, requested outputs, and execution scope. Deterministic CLI code validates the parse and explicit flags. `--input-json` bypasses AI parsing with a complete `OrchestrationRequest`.

## Data Sources and API Calls

The system uses ClinicalTrials.gov API v2 for trial identity, design, arms, interventions, eligibility, endpoints, locations, and analog benchmarking. RxNorm is used for best-effort drug normalization. PubMed E-utilities are used for literature and market-evidence metadata. openFDA drug labels support safety and dosing context. Lens Patent Search is used for patent candidates when `LENS_API_TOKEN` is configured. US Census ACS is used for commercial population denominators when `CENSUS_API_KEY` is configured.

Local source files are first-class inputs. `Source_Based_PoS_Workbook.xlsx` supplies numeric probability of success, `california_wac_data.xlsx` supplies WAC evidence, and YAML configs supply identity rules, market assumptions, query templates, WAC source policy, and rNPV assumptions. Numeric PoS, pricing, commercial forecasts, and rNPV are not AI-generated.

## Entity Resolution

Entity resolution starts from normalized ClinicalTrials.gov records. `asset_identity` selects active, non-placebo interventions, applies RxNorm aliases when available, maps modality and indication through YAML rules, normalizes sponsor aliases, and applies `human_overrides.yaml` when present.

The current trial record remains canonical for NCT-scoped workflows. If asset identity conflicts with current CT.gov trial conditions, commercial pricing, market sizing, peak sales, and rNPV are blocked unless a reviewed human override resolves the conflict.

## Agent 3 Clinical Outcome Prediction

Agent 3 owns `clinical_outcome_prediction`. One NCT ID is fetched from ClinicalTrials.gov, then deterministic components build trial identity, design features, asset identity, historical PoS, comparator landscape, endpoint risk, enrollment/duration risk, safety label context, approval-likelihood proxy, and failure-mode classification.

Bounded manager/subagents may refine interpretation when live agents are enabled. Deterministic fallbacks preserve the same typed output when live agents are disabled or fail. Agent 3 must stay a source-backed clinical risk artifact; it does not make approval, investment, licensing, or go/no-go decisions.

## Agent 4 Due Diligence

Agent 4 owns `due_diligence`. It consumes a typed Agent 3 handoff from Scientific Memory unless `refresh_agent3` forces a new Agent 3 run. It then builds clinical evidence, competitive landscape, openFDA safety summary, Lens patent/LOE review, PoS lookup, pricing evidence, commercial model, rNPV, rule-based red flags, and a draft asset memo.

The commercial model uses reviewed CLI assumptions, source-backed or AI-extracted epidemiology evidence, Census denominators when available, config defaults, pricing evidence, and deterministic patient-funnel/revenue calculations. rNPV is deterministic once commercial revenue, PoS, LOE, launch year, discount rate, tax rate, operating margin, and development cost exist.

## Agent 5 Protocol Design Brief

Agent 5 owns `protocol_design`. It consumes Agent 3 and Agent 4 handoffs, then creates a draft next-study strategy brief anchored in CT.gov analog benchmarking. It is not a full protocol, IRB-ready protocol, submission-ready protocol, enrollment-ready protocol, or final approval artifact.

Agent 5 builds a CT.gov search plan, executes searches deterministically, scores analogs with semantic similarity features, normalizes every candidate disposition, searches same-asset/same-indication/same-sponsor follow-on lineages, calculates benchmark statistics, synthesizes recurring protocol patterns, derives design decisions, drafts protocol sections, and attaches a mandatory human clinical/statistical/regulatory review gate.

## Agent Runtime and Fallbacks

All live reasoning runs through the shared `agent_runtime` layer. It supports OpenAI Agents SDK calls, direct structured LLM calls, route-specific model selection, retries, context compaction, and deterministic fallbacks.

Execution mode is visible everywhere. Workflow outputs, agent traces, agent-output envelopes, Control Tower records, reports, and HTML viewers surface one of `live_agent`, `direct_llm`, `deterministic_fallback`, or `reused_artifact`, plus counts of live AI calls, deterministic fallbacks, and reused artifacts.

## Validation and Human Gates

Every workflow output is validated for schema conformance, source coverage, numeric provenance, and workflow-specific guardrails. Agent 4 also validates consistency with Agent 3. Agent 5 validates draft-only source boundaries, analog benchmark integrity, semantic consistency, support-source labels, duration language, and whole-participant enrollment language.

Human gates are assigned when validators fail, high-risk language appears, important evidence is missing, confidence is low, or the workflow is inherently draft-only. Agent 5 always requires human clinical, statistical, and regulatory review. Failed validation and open gates remain visible in Scientific Memory and reports.

## Output Schema

The output layer is typed and auditable. Workflow outputs contain sources, evidence claims, assumptions, missing-data flags, validation results, confidence flags, human gates, human-readable summaries, execution-mode summaries, and workflow-specific artifacts such as `ClinicalOutcomePredictionOutput`, `DueDiligenceOutput`, `ProtocolDesignOutput`, `AssetMemo`, `CommercialModelOutput`, `RNPVOutput`, `AnalogBenchmarkBundle`, and `ProtocolDesignBrief`.

The report layer builds JSON outputs, run-level HTML viewers, final report payloads, and cumulative NCT development reports from Scientific Memory. Reports summarize source-backed claims, raw artifacts, validation results, confidence flags, gates, execution mode, and reused upstream handoffs.

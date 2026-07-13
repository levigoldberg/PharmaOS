# PharmaOS

AI-native pharma operating system prototype with a memory-aware Control Tower and specialist workflow agents.

Start with:

- [Detailed system architecture guide](docs/system_architecture_detailed.md)

## Architecture

Current orchestration flow:

Objective -> Scientific Memory -> Control Tower -> Capability Registry -> run/reuse/refresh/skip/block -> specialist workflow -> validation/human gates -> memory update -> replan.

The Control Tower reads the objective and a typed Scientific State Snapshot, checks the Capability Registry, and chooses the minimum justified path. The snapshot includes the pending downstream decision, evidence requirements, requirement satisfaction, unresolved or contradictory claims, critical gaps, stale or incompatible artifacts, human gates, and blocked capabilities. The Control Tower reuses compatible artifacts, refreshes stale or invalidated artifacts, skips explicitly skipped capabilities, blocks unavailable modules, and replans after material state changes.

Planning is decision-aware rather than keyword-only. For example, a Phase II -> Phase III decision can reuse fresh Agent 3 and Agent 4 artifacts, identify missing Phase III analog evidence, run Agent 5 only when justified, then stop for human review. Deterministic validation checks dependency order, reuse validity, stale/invalid artifacts, human gates, unavailable skeleton modules, and whether each run/refresh step addresses decision evidence requirements.

AI execution mode is explicit in workflow outputs, agent traces, agent-output envelopes, Control Tower records, and reports. The supported modes are `live_agent`, `direct_llm`, `deterministic_fallback`, and `reused_artifact`; reports summarize counts such as live AI calls completed and deterministic fallbacks used.

Live model calls use shared context compaction before sending payloads to the Responses API or Agents SDK. Full typed state remains in Scientific Memory and output artifacts; only the model-facing JSON is bounded. Agent traces record whether compaction was applied and how much context was trimmed.

Implemented executable workflows:

- Agent 3 `clinical_outcome_prediction`: clinical outcome/risk context for one NCT ID, with deterministic trial identity, design, PoS, safety, and trial-landscape components plus SDK-backed bounded reasoning when live agents are enabled.
- Agent 4 `due_diligence`: clinical-stage diligence using Agent 3 handoff plus CT.gov, PubMed, openFDA labels, Lens when configured, local PoS/WAC workbooks, commercial modeling, and rNPV.
- Agent 5 `protocol_design`: draft next-study protocol design brief using Agent 3/4 handoffs, CT.gov analog benchmarking, PubMed/openFDA context, and local templates/checklists.

Registered non-executable skeletons:

- `discovery`
- `tox_pkpd_safety`
- `enrollment_feasibility`
- `trial_execution`
- `manufacturing_biofactory`
- `launch_pv`
- `regulatory_quality_audit`

Skeleton capabilities are visible to the Scientific State and Control Tower so they can be planned against and safely blocked with missing connectors instead of executed or invented.

## Environment

Install the repo into the virtual environment so `python -m pharma_os` and the `pharma-os` console script expose the current CLI:

```bash
./.venv/bin/python -m pip install ".[dev]"
```

Copy `.env.example` to `.env` and fill only the keys you have.

Required for live agent runs:

- `OPENAI_API_KEY`
- `PHARMA_OS_MODEL`, optional global model override. If unset, PharmaOS uses route-specific defaults.
- Route-specific model overrides:
  `PHARMA_OS_MODEL_REQUEST_UNDERSTANDING`,
  `PHARMA_OS_MODEL_CONTROL_TOWER`,
  `PHARMA_OS_MODEL_HUMAN_SUMMARY`,
  `PHARMA_OS_MODEL_AGENT3_MANAGER`,
  `PHARMA_OS_MODEL_AGENT3_SUBAGENT`,
  `PHARMA_OS_MODEL_AGENT4_MANAGER`,
  `PHARMA_OS_MODEL_AGENT4_SUBAGENT`,
  `PHARMA_OS_MODEL_AGENT5_MANAGER`,
  `PHARMA_OS_MODEL_AGENT5_SUBAGENT`.
- Default route tiers when no override is set: fast routes use `gpt-5.6-luna`, balanced routes use `gpt-5.6-terra`, and deep protocol-writing routes use `gpt-5.6-sol`.
- `PHARMA_OS_ENABLE_LIVE_AGENTS=false` forces deterministic offline fallbacks even when an API key exists.
- `PHARMA_OS_AGENTS_DISABLED=true` or `PHARMA_OS_OFFLINE=true` also forces deterministic offline fallbacks.
- `PHARMA_OS_AGENT_MAX_TURNS`, defaults to `8`
- `PHARMA_OS_LLM_MAX_RETRIES`, defaults to `4`; transient rate limits, timeouts, and server errors retry before deterministic fallback.
- `PHARMA_OS_LLM_RETRY_INITIAL_DELAY_SECONDS`, defaults to `1.0`
- `PHARMA_OS_LLM_RETRY_MAX_DELAY_SECONDS`, defaults to `30.0`
- `PHARMA_OS_LLM_MAX_INPUT_CHARS`, defaults to `60000`; maximum model-facing JSON payload size before hard compaction.
- `PHARMA_OS_LLM_MAX_STRING_CHARS`, defaults to `2500`; maximum per-string length before recursive trimming.
- `PHARMA_OS_LLM_MAX_ARRAY_ITEMS`, defaults to `30`; maximum per-array item count before recursive trimming.
- `PHARMA_OS_LLM_MAX_JSON_DEPTH`, defaults to `10`; maximum nested JSON depth before summarization.
- `PHARMA_OS_LLM_CONTEXT_COMPACTION_DISABLED=true` disables model-facing compaction for debugging only.

Optional for due diligence:

- `NCBI_API_KEY` and `NCBI_EMAIL` for more reliable PubMed market-evidence retrieval
- `CENSUS_API_KEY` for live US Census ACS population denominators used in prevalence-to-patient conversion; without it, Agent 4 uses the reviewed 2024 ACS config fallback and flags human review
- `PHARMA_OS_CENSUS_YEAR` to pin the Census ACS year instead of auto-discovery
- `PHARMA_OS_PUBMED_MAX_RETRIES`, `PHARMA_OS_PUBMED_RETRY_INITIAL_DELAY_SECONDS`, and `PHARMA_OS_PUBMED_RETRY_MAX_DELAY_SECONDS` for PubMed transient retry behavior
- `PHARMA_OS_MARKET_MAX_QUERIES` and `PHARMA_OS_MARKET_PUBMED_RESULTS_PER_QUERY` to tune Agent 4 market-evidence breadth
- `LENS_API_TOKEN` for Lens patent retrieval
- `PHARMA_OS_POS_WORKBOOK_PATH`, defaults to `data/Source_Based_PoS_Workbook.xlsx`
- `PHARMA_OS_WAC_DATA_PATH`, defaults to `data/california_wac_data.xlsx`

Config layout:

- Shared identity rules live in `src/pharma_os/data/shared/`.
- Due-diligence assumptions and source registries live in `src/pharma_os/data/due_diligence/`.
- Due diligence applies values in this order: source-backed or calculated values, user-reviewed CLI input, config fallback, then a missing-data flag plus human gate.

## Example Commands

Control Tower orchestration:

```bash
python -m pharma_os orchestrate \
--goal "Force refresh the whole suite of agents: trial prediction, commercial due diligence, and protocol design for NCT05966480"
```

Natural-language orchestration uses the AI request-understanding step for workflow selection and identifier extraction from `--goal`; deterministic code only validates the AI parse and protects execution. Explicit fields such as `--nct-id` override or validate the AI-extracted fields, but they do not skip AI request understanding. Use `--input-json` with a complete `OrchestrationRequest` when you intentionally want a fully structured non-AI request. If `--output-json` and `--output-html` are omitted, JSON and HTML reports are written under `outputs/`.

Clinical outcome prediction:

```bash
python -m pharma_os run clinical_outcome_prediction \
  --nct-id NCT04903795 \
  --db-path .pharma_os/scientific_memory.sqlite \
  --output-json outputs/clinical_outcome_prediction.json
```

Due diligence:

```bash
python -m pharma_os run due_diligence \
  --nct-id NCT04903795 \
  --annual-patients 1000 \
  --peak-penetration 0.2 \
  --gross-to-net 0.15 \
  --operating-margin 0.35 \
  --discount-rate 0.1 \
  --development-cost 50000000 \
  --launch-year 2029 \
  --loe-year 2040 \
  --db-path .pharma_os/scientific_memory.sqlite \
  --output-json outputs/due_diligence.json
```

Protocol design brief:

```bash
python -m pharma_os run protocol_design \
  --nct-id NCT04903795 \
  --db-path .pharma_os/scientific_memory.sqlite \
  --output-json outputs/protocol_design.json
```

Bounded smoke test with public APIs and deterministic agent fallbacks:

```bash
PHARMA_OS_AGENTS_DISABLED=true python -m pharma_os run clinical_outcome_prediction \
  --nct-id NCT04903795 \
  --db-path /tmp/pharma_os_agent3.sqlite \
  --output-json /tmp/pharma_os_agent3.json \
  --output-html /tmp/pharma_os_agent3.html
```

Report from Scientific Memory:

```bash
python -m pharma_os report \
  --run-id RUN_ID \
  --db-path .pharma_os/scientific_memory.sqlite \
  --output-json outputs/report.json
```

HTML run viewer:

```bash
python -m pharma_os view \
  --run-id RUN_ID \
  --db-path .pharma_os/scientific_memory.sqlite \
  --output-html outputs/run_view.html
```

Notes:

- `trial_intelligence` remains a compatibility route over the internal Agent 3 trial-landscape component; it is not a separate top-level LLM agent.
- The workflows do not invent LOE, PoS, pricing, market size, rNPV, protocol, regulatory, or clinical decision inputs. Missing or low-confidence inputs create confidence flags and human gates.
- Agent 5 returns a draft `ProtocolDesignBrief` requiring human review. It is not a full protocol, IRB-ready protocol, submission-ready protocol, or final decision.

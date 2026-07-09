# PharmaOS

AI-native pharma operating system prototype.

Start with:

- [Project brief](docs/project_brief.md)
- [Architecture decisions](docs/architecture_decisions.md)
- [Lean implementation plan](docs/build_plan.md)

## Environment

Copy `.env.example` to `.env` and fill only the keys you have.

Required for live agent runs:

- `OPENAI_API_KEY`
- `PHARMA_OS_MODEL`, defaults to `gpt-5.5`
- Live Agent 3/4/5 subagents call the OpenAI Agents SDK when `OPENAI_API_KEY` is present.
- `PHARMA_OS_ENABLE_LIVE_AGENTS=false` forces deterministic offline fallbacks even when an API key exists.
- `PHARMA_OS_AGENTS_DISABLED=true` or `PHARMA_OS_OFFLINE=true` also forces deterministic offline fallbacks.
- `PHARMA_OS_AGENT_MAX_TURNS`, defaults to `8`

Optional for due diligence:

- `LENS_API_TOKEN` for Lens patent retrieval
- `PHARMA_OS_POS_WORKBOOK_PATH`, defaults to `data/Source_Based_PoS_Workbook.xlsx`
- `PHARMA_OS_WAC_DATA_PATH`, defaults to `data/california_wac_data.xlsx`

Config layout:

- Shared identity rules live in `src/pharma_os/data/shared/`.
- Due-diligence assumptions and source registries live in `src/pharma_os/data/due_diligence/`.
- Due diligence applies values in this order: source-backed or calculated values, user-reviewed CLI input, config fallback, then a missing-data flag plus human gate.
- Config fallback values are persisted as `AssumptionRecord` objects with config source IDs and filename/field-path provenance.

## Example Commands

Clinical outcome prediction, the canonical Agent 3 workflow:

```bash
python -m pharma_os run clinical_outcome_prediction \
  --nct-id NCT04903795 \
  --db-path .pharma_os/scientific_memory.sqlite \
  --output-json outputs/clinical_outcome_prediction.json
```

Agent 3 trial-landscape compatibility mode:

```bash
python -m pharma_os run trial_intelligence \
  --disease "glioblastoma" \
  --target EGFR \
  --limit 10 \
  --db-path .pharma_os/scientific_memory.sqlite \
  --output-json outputs/trial_intelligence.json
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

Protocol design brief, the Agent 5 draft-strategy workflow:

```bash
python -m pharma_os run protocol_design \
  --nct-id NCT04903795 \
  --db-path .pharma_os/scientific_memory.sqlite \
  --output-json outputs/protocol_design.json
```

Report from Scientific Memory:

```bash
python -m pharma_os report \
  --run-id RUN_ID \
  --db-path .pharma_os/scientific_memory.sqlite \
  --output-json outputs/report.json
```

Notes:

- `trial_intelligence` is a thin compatibility route over the internal Agent 3 trial-landscape component used by `clinical_outcome_prediction`; it is not a separate LLM agent.
- Due diligence uses ClinicalTrials.gov, RxNorm, local PoS/WAC workbooks, openFDA labels, and Lens when `LENS_API_TOKEN` is valid.
- Due diligence remains the Agent 4 owner for patents, pricing, commercial model, market sizing, BD logic, and rNPV.
- The workflow does not invent LOE, PoS, pricing, market size, or rNPV inputs. Missing or low-confidence inputs create confidence flags and human gates.
- `protocol_design` consumes Agent 3 and Agent 4 outputs, then uses CT.gov analog trials, PubMed metadata/abstract context, openFDA label context, and local templates/checklists only.
- Agent 5 returns a typed `ProtocolDesignBrief` with an `AnalogBenchmarkBundle`; it is a draft strategy artifact requiring human review, not a full protocol, IRB-ready protocol, submission-ready protocol, or final decision.

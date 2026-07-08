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

Optional for due diligence:

- `LENS_API_TOKEN` for Lens patent retrieval
- `PHARMA_OS_POS_WORKBOOK_PATH`, defaults to `data/Source_Based_PoS_Workbook.xlsx`
- `PHARMA_OS_WAC_DATA_PATH`, defaults to `data/california_wac_data.xlsx`

## Example Commands

Clinical trial intelligence:

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

Report from Scientific Memory:

```bash
python -m pharma_os report \
  --run-id RUN_ID \
  --db-path .pharma_os/scientific_memory.sqlite \
  --output-json outputs/report.json
```

Notes:

- Due diligence uses ClinicalTrials.gov, RxNorm, local PoS/WAC workbooks, openFDA labels, and Lens when `LENS_API_TOKEN` is valid.
- The workflow does not invent LOE, PoS, pricing, market size, or rNPV inputs. Missing or low-confidence inputs create confidence flags and human gates.

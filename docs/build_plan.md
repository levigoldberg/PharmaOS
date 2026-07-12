# AI-Native PharmaOS Lean Implementation Plan

Status: this is the historical lean build plan. The current implementation state is documented in `README.md`, `AGENTS.md`, `docs/architecture_decisions.md`, `docs/system_architecture_detailed.md`, and `docs/system_architecture_concise.md`: Agent 3 `clinical_outcome_prediction`, Agent 4 `due_diligence`, and Agent 5 `protocol_design` are executable; discovery, tox/PK-PD, enrollment, trial execution, manufacturing, launch/PV, and regulatory/quality/audit are registered non-executable skeletons.

## 1. Recommended repo structure

Keep `src/` and remove `src2/`. `src/` contains the strict shared schemas that match `AGENTS.md`; `src2/` is a conflicted duplicate with `.DS_Store` files and merge markers in `schemas.py`.

Recommended structure:

```text
PharmaOS/
  AGENTS.md
  README.md
  docs/
    build_plan.md
    project_brief.md
    architecture_decisions.md
  src/pharma_os/
    cli.py
    orchestrator.py
    schemas.py
    memory.py
    validators.py
    report.py
    agents/
      discovery.py
      safety.py
      trial_intel.py
      protocol.py
      launch_pv.py
      biofactory.py
    tools/
      pubmed.py
      pubchem.py
      chembl.py
      clinicaltrials_gov.py
      openfda.py
      dailymed.py
      rxnorm.py
      calculators.py
    workflows/
      discovery_landscape.py
      tox_pkpd_safety.py
      trial_due_diligence.py
      protocol_feasibility.py
      launch_rwe_pv.py
      biofactory.py
  tests/
```

Do not build around a single asset identifier. Use workflow-specific input objects: target/gene, molecule/SMILES, disease/indication, NCT ID, protocol concept, label/drug name, or manufacturing modality. NCT IDs are one valid input for trial intelligence, not the project center.

## 2. Framework/tool decisions

- Orchestration: OpenAI Agents SDK, but keep Python workflow functions as the control plane.
- Schemas: Pydantic v2 strict models for all inputs, tool outputs, agent outputs, validation results, memory writes, and reports.
- Persistence: SQLite Scientific Memory first. No vector DB until retrieval quality requires chunk search.
- HTTP: `httpx` with timeouts, explicit error objects, typed normalized results, and source metadata.
- Tests: `pytest`, fixtures for API payloads, no network dependency in unit tests.
- Interface: CLI first: `python -m pharma_os run <workflow> ...` and `python -m pharma_os report --run-id <id>`.
- Avoid for now: LangChain, LangGraph, FastAPI, React, Docker, Neo4j, Postgres, and broad dashboards.
- Optional science packages by slice: RDKit for molecule descriptors; DeepChem/TDC only after a narrow endpoint is chosen; BioSTEAM/Pyomo/BoTorch only if the BioFactory slice has a real public dataset.

## 3. Workflow-by-workflow plan

### Discovery / target landscape

- Input: disease/indication plus optional target gene, pathway, modality, or molecule.
- Real APIs/tools: PubMed E-utilities, Europe PMC, ChEMBL, PubChem, DepMap, GDC/TCGA, PDB/RCSB, OpenTargets if added, RDKit for molecule-level descriptors.
- Agents: target landscape agent, evidence synthesis agent, novelty/IP triage agent.
- Deterministic tools: literature search, target/drug lookup, bioactivity retrieval, dependency evidence retrieval, molecule descriptor calculator, source normalizer.
- Output schema: `TargetLandscapeOutput` with targets, mechanisms, modalities, evidence claims, tractability flags, novelty flags, and source IDs.
- Validators: source coverage per target, duplicate target/entity normalization, unsupported claim detector, numeric provenance for assay/bioactivity values.
- Scientific Memory writes: target entities, disease entities, sources, claims, agent output, validation results, human gates.
- Human gates: final target nomination, lead-series selection, IND-candidate nomination.
- Build now: target/drug/literature landscape for one indication. Defer molecule generation, docking, active learning, and robotic validation planning.

### Tox / PKPD / safety screen

- Input: molecule name, SMILES/InChI, known drug, or investigational asset identity.
- Real APIs/tools: PubChem, ChEMBL, openFDA labels, openFDA FAERS, DailyMed, Tox21/ToxCast where accessible, RDKit descriptors. PK-DB and Open Systems Pharmacology are candidates for later deeper PK/PBPK work.
- Agents: safety evidence agent, tox risk agent, dose rationale critic.
- Deterministic tools: compound normalization, label safety extraction, FAERS query, known-target/off-target lookup, descriptor calculation, simple risk scoring.
- Output schema: `SafetyScreenOutput` with known safety issues, label warnings, FAERS signals, assay/tox evidence, DDI/hERG/hepatotoxicity flags if sourced, and dose-not-ready status.
- Validators: every safety claim has source IDs, FAERS disproportionality labeled as signal not causality, no first-in-human dose recommendation without human gate.
- Scientific Memory writes: molecule entities, adverse event claims, label sources, FAERS query metadata, safety flags, validation results.
- Human gates: IND-enabling tox strategy, first-in-human dose, dose escalation, stopping rules.
- Build now: source-backed safety screen and "not dose-decision ready" output. Defer PBPK, PopPK, human dose scenarios, and regulated tox package decisions.

### Trial intelligence + due diligence

- Input: NCT ID, asset name, sponsor, indication, target, or diligence memo seed.
- Real APIs/tools: ClinicalTrials.gov, PubMed, Europe PMC, RxNorm, openFDA labels/FAERS, Drugs@FDA, Orange Book files, EMA medicine data, SEC EDGAR, patent sources if API credentials exist, plus deterministic rNPV/commercial calculators.
- Agents: trial evidence agent, due-diligence red-flag agent, patent/LOE critic, commercial/rNPV critic.
- Deterministic tools: CT.gov client, asset identity parser, PubMed retrieval, RxNorm normalization, openFDA/DailyMed label lookup, patent search adapter, PoS workbook lookup, pricing/rNPV calculators.
- Output schema: `DueDiligenceOutput` with asset identity, trial landscape, efficacy/safety evidence, competitive context, IP/LOE status, PoS, valuation assumptions, red flags, and source-backed confidence.
- Validators: NCT format only when NCT input is used, source coverage by section, numeric provenance for PoS/pricing/rNPV, stale/failed API flags, cross-check asset names across sources.
- Scientific Memory writes: asset, trials, sponsors, interventions, papers, labels, patents, claims, calculator assumptions, red flags, human gates.
- Human gates: go/no-go, license/acquire/invest, patent-family reliance, final valuation.
- Build now: a thin workflow that can accept NCT or asset input and reuse selected tools later. Defer full old-repo port until shared memory, validation, and report contracts are stable.

### Protocol + enrollment feasibility

- Input: asset strategy, indication, phase, endpoint concept, target population, comparator, geography.
- Real APIs/tools: ClinicalTrials.gov trial comparators, PubMed, ICH M11, ICH E6(R3), FDA/EMA guidance pages, CDISC terminology, Criteria2Query/OHDSI concepts later.
- Agents: protocol design agent, eligibility burden agent, enrollment feasibility agent, regulatory consistency critic.
- Deterministic tools: comparator trial retrieval, endpoint/eligibility extraction, criteria complexity scorer, site/country distribution summaries from public trials.
- Output schema: `ProtocolFeasibilityOutput` with draft synopsis, eligibility criteria, endpoint rationale, comparator evidence, enrollment risks, site/geography recommendations, and review gates.
- Validators: no patient matching without real consented/EHR data, source-backed protocol rationale, eligibility criteria parseability, human approval required for protocol.
- Scientific Memory writes: protocol concept, endpoint claims, eligibility criteria, comparable trial sources, feasibility flags, gates.
- Human gates: final protocol approval, patient contact/consent, final eligibility, regulatory commitments.
- Build now: protocol synopsis plus public-trial feasibility benchmark. Defer EHR/OMOP patient matching, outreach, scheduling, and trial execution systems.

### Launch / RWE / pharmacovigilance

- Input: approved drug/label, indication, comparator set, geography, launch question.
- Real APIs/tools: openFDA FAERS, DailyMed SPL, FDA labels, Orange Book, CMS Part D prescribing data, Open Payments, FDA RWE resources.
- Agents: label/market evidence agent, PV signal triage agent, RWE hypothesis agent, promotional compliance critic.
- Deterministic tools: label parser, FAERS event query, CMS prescribing query, Open Payments query, safety signal summarizer.
- Output schema: `LaunchPvOutput` with label summary, safety signals, prescriber/payment context, RWE questions, pharmacovigilance flags, and compliant/non-promotional claim status.
- Validators: FAERS signals not causal, promotional claims blocked, all label claims source-backed, material safety issues require human review.
- Scientific Memory writes: labels, adverse event claims, RWE hypotheses, PV flags, prescriber/payment sources, validation results.
- Human gates: serious safety decisions, label expansion, medical/legal/regulatory review, promotional claims.
- Build now: label plus FAERS safety dashboard in CLI/report form. Defer claims/EHR outcomes, payer/formulary data, CRM, field-force workflows.

### BioFactory

- Input: modality, product class, process question, or public fermentation/process dataset.
- Real APIs/tools: BioSTEAM, Pyomo, BoTorch, PenSimPy, public fermentation/bioreactor sample datasets if usable.
- Agents: process feasibility agent, CMC risk critic.
- Deterministic tools: dataset loader, simple process simulator, optimization stub, `not_implemented` connector result when data is missing.
- Output schema: `BioFactoryOutput | NotImplementedOutput`.
- Validators: block GMP claims, batch release, deviation closure, and physical process-control actions unless connected to regulated systems and human approval.
- Scientific Memory writes: process assumptions, dataset sources, simulation outputs, validation results, GMP gates.
- Human gates: GMP batch release, deviations, comparability, CMC regulatory commitments.
- Build now only if a real public dataset can support a toy process-development slice. Otherwise return typed `not_implemented`. Defer MES/LIMS/QMS/PAT hardware, batch records, and closed-loop control.

## 4. Scientific Memory design

Use SQLite tables:

- `runs`: run id, workflow, status, input JSON, timestamps.
- `sources`: source id, type, URL, title, retrieved_at, checksum/version, raw pointer.
- `entities`: entity id, entity type, canonical name, aliases, external IDs.
- `claims`: claim id, text, entity links, source ids, confidence, qualifiers, validation status.
- `agent_outputs`: output id, run id, agent name, typed JSON, source ids, confidence.
- `validation_results`: validation id, target id, status, message, validator, confidence.
- `human_gates`: gate id, run id, decision, reason, required roles, reviewer, reviewed_at.
- `reports`: report id, run id, title, summary, typed JSON, validation status.

Store raw API payloads as JSON blobs or filesystem artifacts referenced by `sources`. Store no unsupported prose as fact. Every claim must point to source IDs or be explicitly marked as an agent hypothesis.

## 5. Validation and human-gate design

Validation should be deterministic and visible in reports:

- schema validation for every typed object
- source coverage checks for every claim and report section
- numeric provenance checks for PoS, prices, exposure, rates, counts, NPV, and confidence scores
- cross-source consistency checks for asset names, sponsors, indications, labels, trial phase/status, and dates
- uncertainty flags for missing, stale, failed, or conflicting data
- human-gate assignment based on risk category, not just low confidence

Confidence should be a structured result, not a vibes score: `confidence`, `confidence_level`, `confidence_flags`, `validation_status`, and `gate_reason`.

## 6. Reuse plan for `levigoldberg/ClinicalTrialIntel`

Reuse later for the Trial Intelligence / Due Diligence workflow only:

- Port API clients to `tools/` and convert `requests` to `httpx`.
- Keep deterministic parsers/calculators: CT.gov, RxNorm, PubMed/Europe PMC, asset identity, PoS workbook lookup, pricing, commercial forecast, rNPV, patent/LOE adapters.
- Convert old Pydantic models into workflow-specific schemas that compose the shared PharmaOS provenance, claims, validation, confidence, and gate models.
- Keep useful tests and fixtures, especially network-independent parser tests.
- Do not port the old pipeline orchestrator as the global architecture.
- Do not make NCT ID the universal entrypoint. Trial/diligence can accept NCT ID, but other workflows use target, molecule, disease, protocol, drug, label, or modality inputs.
- Preserve old repo behavior as reference fixtures before refactoring.

## 7. Deferred modules and why

- Trial execution: needs EDC, CTMS, eTMF, RBQM, monitoring, deviations, and regulated study operations data.
- Patient matching/enrollment execution: needs consented EHR/claims/OMOP/FHIR data, site agreements, recruitment systems, and human consent workflows.
- PBPK/PopPK and first-in-human dose: needs validated models, preclinical data, clinical pharmacology review, and regulatory-grade workflows.
- GMP BioFactory operations: needs MES, LIMS, QMS, eBR, PAT streams, process equipment, validated control systems, and QA release authority.
- Launch commercial execution: needs claims/EHR, payer/formulary data, CRM, MLR workflows, and promotional compliance systems.
- Regulatory submissions/audit layer: should start as traceability and report output, not full submission automation.

## 8. First implementation tasks for Codex

1. Make `MemoryStore` SQLite-backed with migrations for runs, sources, claims, outputs, validations, gates, and reports.
2. Add workflow input/output schemas that compose the shared provenance, evidence, validation, confidence, and human-gate primitives.
3. Implement deterministic `httpx` clients for PubMed, PubChem, ClinicalTrials.gov, openFDA, DailyMed, ChEMBL, and RxNorm with typed responses and fixtures.
4. Implement validators for schema validity, source coverage, numeric provenance, cross-source consistency, confidence flags, and human-gate assignment.
5. Implement the first real slice: discovery target landscape for one disease/target input.
6. Add report rendering that surfaces sources, failed validators, confidence flags, and human gates.
7. Add trial/diligence workflow skeleton, then selectively adapt old ClinicalTrialIntel components into tools.
8. Add safety and launch/PV slices with openFDA/DailyMed/FAERS data.
9. Add protocol feasibility using public trials and guidance sources.
10. Add BioFactory only as typed `not_implemented` unless a public dataset supports a real simulation slice.

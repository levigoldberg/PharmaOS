# Project Brief

Build an AI-native pharma operating system prototype.

The project should demonstrate how a vertically integrated AI-native pharma company could coordinate specialized agents across the drug-development lifecycle with as much automation as safely possible.

The system should include these layers:

1. Scientific Memory
2. Agent Orchestrator / Control Tower
3. Deterministic API and calculation tools
4. Specialist agents
5. Validation and confidence scoring
6. Human-gate logic
7. Final report / audit output

The goal is not one single pipeline. The goal is multiple working vertical slices that share the same architecture.

Each vertical slice should follow this pattern:

input
-> real data/API/tool retrieval
-> specialist agent reasoning
-> typed output
-> validation/scoring
-> Scientific Memory write
-> report output

Priority working slices:

1. Discovery / target landscape
2. Tox / PKPD / safety screen
3. Trial intelligence + due diligence
4. Protocol + enrollment feasibility
5. Launch / RWE / pharmacovigilance
6. BioFactory only if real public tools/data make it feasible

Use real APIs and open-source tools wherever possible.

Preferred data/tools:

- ClinicalTrials.gov
- PubMed E-utilities
- openFDA labels
- openFDA FAERS
- PubChem
- DailyMed
- ChEMBL
- RxNorm
- FDA / EMA / ICH regulatory documents
- RDKit
- DeepChem or Therapeutics Data Commons if feasible
- BioSTEAM, Pyomo, BoTorch, or PenSimPy for BioFactory if feasible

Do not create fake patient, site, batch, EHR, CTMS, EDC, MES, LIMS, or manufacturing data by default. If a module requires unavailable regulated data or physical systems, return a typed not_implemented result with the missing connector or required system.

Automate by default. Human gates should be reserved for decisions that require human accountability: final target nomination, IND-enabling tox strategy, first-in-human dose, final protocol approval, patient contact/consent, final eligibility, serious safety decisions, GMP batch release, regulatory commitments, promotional claims, label expansion, and major capital allocation.

## Existing reusable repository

There is an existing repository that should be inspected before implementing the Trial Intelligence / Due Diligence vertical slice:

- `levigoldberg/ClinicalTrialIntel`

Use it as reusable source material, not as the governing architecture for this new project.

Relevant reusable components likely include:

- ClinicalTrials.gov API client
- RxNorm normalization
- PubMed / Europe PMC evidence retrieval
- asset identity logic
- modality / indication / sponsor rule files
- patent / LOE workflow
- PoS workbook classification and deterministic lookup
- pricing analog workflow and openFDA label dosing support
- commercial model calculator
- rNPV calculator
- final report aggregation patterns
- tests and CLI patterns

Important constraint:
The new project is a broader AI-native pharma operating system with multiple vertical slices. Do not make the whole new project NCT-centered just because the existing repo is NCT-centered.

Use the existing repo later when implementing the Trial Intelligence / Due Diligence workflow. First build the new project's shared architecture: workflows, agents, tools, schemas, Scientific Memory, validators, and report/audit output.

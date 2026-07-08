"""Local source-workbook probability-of-success lookup."""

from __future__ import annotations

import os
from pathlib import Path

from openpyxl import load_workbook

from pharma_os.schemas import AssetIdentityOutput, ClinicalTrialRecord, MissingDataFlag, PoSOutput, SourceMetadata
from pharma_os.tools._due_diligence_common import DEFAULT_POS_WORKBOOK, json_scalar, missing, norm, slug, to_float


def lookup_pos(
    trial: ClinicalTrialRecord,
    asset: AssetIdentityOutput,
    *,
    workbook_path: str | None = None,
) -> tuple[PoSOutput, SourceMetadata]:
    """Lookup source-only PoS from the local workbook."""

    path = Path(workbook_path or os.getenv("PHARMA_OS_POS_WORKBOOK_PATH") or DEFAULT_POS_WORKBOOK)
    source = SourceMetadata(
        source_id=f"pos_workbook:{slug(path.name)}",
        title="Source-based probability of success workbook",
        url=None,
        provenance="Local Source_Based_PoS_Workbook.xlsx AllBenchmarks lookup",
        source_type="pos_workbook",
        version=path.name,
    )
    flags: list[MissingDataFlag] = []
    if not path.exists():
        return (
            PoSOutput(workbook_path=str(path), source_ids=(), missing_data_flags=(missing("pos-workbook-missing", "pos", "workbook_path", f"Workbook not found: {path}", "high"),), confidence=0.0),
            source,
        )
    phase = _phase_for_workbook(trial.phases)
    disease_area = _disease_area_for_workbook(asset.therapeutic_area, trial.conditions)
    if not phase:
        flags.append(missing("pos-phase-missing", "pos", "current_phase", "Trial phase could not be mapped to workbook labels.", "high"))
    if not disease_area:
        flags.append(missing("pos-disease-area-missing", "pos", "disease_area", "Disease area could not be mapped to workbook labels.", "high"))
    value = None
    row_out: dict[str, str | int | float | bool | None] = {}
    lookup_key = f"Disease Area|{disease_area}|{phase}" if disease_area and phase else None
    if lookup_key:
        try:
            wb = load_workbook(path, data_only=False, read_only=False)
            ws = wb["AllBenchmarks"]
            headers = [str(cell.value).strip() for cell in next(ws.iter_rows(min_row=1, max_row=1))]
            for row in ws.iter_rows(min_row=2, values_only=True):
                row_map = dict(zip(headers, row))
                if norm(str(row_map.get("Key") or "")) == norm(lookup_key):
                    value = to_float(row_map.get("Phase LOA"))
                    row_out = {str(k): json_scalar(v) for k, v in row_map.items() if k is not None}
                    break
            if value is None:
                flags.append(missing("pos-row-missing", "pos", "probability_of_success", f"No workbook row found for {lookup_key}.", "high"))
        except Exception as exc:
            flags.append(missing("pos-workbook-error", "pos", "probability_of_success", f"Workbook lookup failed: {exc.__class__.__name__}.", "high"))
    return (
        PoSOutput(
            probability_of_success=value,
            current_phase=phase,
            disease_area=disease_area,
            workbook_path=str(path),
            lookup_key=lookup_key,
            benchmark_row=row_out,
            source_ids=(source.source_id,) if value is not None else (),
            missing_data_flags=tuple(flags),
            confidence=0.9 if value is not None else 0.0,
        ),
        source,
    )


def _disease_area_for_workbook(therapeutic_area: str | None, conditions: tuple[str, ...]) -> str | None:
    text = " ".join([therapeutic_area or "", *conditions]).casefold()
    if "oncology" in text or "cancer" in text or "tumor" in text or "glioblastoma" in text:
        return "Oncology"
    if "neurology" in text or "alzheimer" in text or "parkinson" in text:
        return "Neurology"
    if "immunology" in text:
        return "Autoimmune"
    return None


def _phase_for_workbook(phases: tuple[str, ...]) -> str | None:
    text = " ".join(phases).upper()
    if "PHASE3" in text or "PHASE 3" in text:
        return "Phase III"
    if "PHASE2" in text or "PHASE 2" in text:
        return "Phase II"
    if "PHASE1" in text or "PHASE 1" in text or "EARLY_PHASE1" in text:
        return "Phase I"
    return None

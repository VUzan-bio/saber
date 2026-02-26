#!/usr/bin/env python3
"""Design crRNA candidates for the complete MDR-TB 14-plex resistance panel.

Runs SABER across all WHO-critical mutations covering first-line (RIF, INH,
EMB, PZA) and second-line (FLQ, AMK/KAN) drugs.  Handles all mutation types:
amino-acid substitutions, promoter SNPs, and rRNA point mutations.

Usage (from repo root):
    python scripts/design_core_panel.py \
        --reference data/references/H37Rv.fasta \
        --gff data/references/H37Rv.gff3 \
        --output results/mdr_14plex

Outputs per target:
    results/mdr_14plex/<gene>_<mutation>/targets.json
    results/mdr_14plex/<gene>_<mutation>/candidates.json
    results/mdr_14plex/<gene>_<mutation>/scored_candidates.json
    results/mdr_14plex/panel_summary.json       ← aggregated top candidates
    results/mdr_14plex/panel_summary.tsv         ← tab-delimited for Excel
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure saber is importable when run from scripts/ or repo root
# ---------------------------------------------------------------------------
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from saber.core.types import Drug, Mutation
from saber.core.config import PipelineConfig, ReferenceConfig
from saber.pipeline.runner import SABERPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("design_core_panel")

# ======================================================================
# WHO Critical Mutation Panel — 14-plex MDR/pre-XDR TB
# ======================================================================
# Each entry: (gene, mutation_notation, drug, who_confidence, notes)
#
# Mutation notation follows WHO catalogue conventions:
#   - Amino acid:  S531L, H526Y, D516V, S315T, M306V, D94G, A90V, H57D
#   - Promoter:    c.-15C>T, c.-14C>T  (upstream of gene start)
#   - rRNA:        A1401G, A1484T      (nucleotide position on rRNA gene)
#
# The resolver handles E. coli→M.tb renumbering automatically.
# ======================================================================

# fmt: off
PANEL_MUTATIONS: list[dict] = [
    # ── First-line: Rifampicin (RIF) — rpoB RRDR ──
    {"gene": "rpoB",  "mutation": "S531L",    "drug": "RIF", "confidence": "assoc w resistance",
     "notes": "Most common RIF mutation globally (~40-70% of RIF-R). E.coli numbering; M.tb=S450L."},
    {"gene": "rpoB",  "mutation": "H526Y",    "drug": "RIF", "confidence": "assoc w resistance",
     "notes": "Second most common RIF mutation (~10-25%). E.coli=H526Y, M.tb=H445Y."},
    {"gene": "rpoB",  "mutation": "D516V",    "drug": "RIF", "confidence": "assoc w resistance",
     "notes": "Third most common RIF mutation (~5-10%). E.coli=D516V, M.tb=D435V."},

    # ── First-line: Isoniazid (INH) — katG + inhA promoter ──
    {"gene": "katG",  "mutation": "S315T",    "drug": "INH", "confidence": "assoc w resistance",
     "notes": "Most common INH mutation globally (~50-90% of INH-R). Abolishes catalase-peroxidase activation."},
    {"gene": "fabG1", "mutation": "c.-15C>T", "drug": "INH", "confidence": "assoc w resistance",
     "notes": "inhA promoter mutation. Uses fabG1/inhA operon. ~15-35% of INH-R. Low-level resistance."},

    # ── First-line: Ethambutol (EMB) — embB ──
    {"gene": "embB",  "mutation": "M306V",    "drug": "EMB", "confidence": "assoc w resistance",
     "notes": "Most common EMB mutation (~40-65%). Codon 306 is the hotspot."},
    {"gene": "embB",  "mutation": "M306I",    "drug": "EMB", "confidence": "assoc w resistance",
     "notes": "Second allele at codon 306 (~10-20% of EMB-R)."},

    # ── First-line: Pyrazinamide (PZA) — pncA ──
    {"gene": "pncA",  "mutation": "H57D",     "drug": "PZA", "confidence": "assoc w resistance",
     "notes": "Common pncA mutation. pncA is highly polymorphic — >300 mutations reported."},
    {"gene": "pncA",  "mutation": "D49N",     "drug": "PZA", "confidence": "assoc w resistance",
     "notes": "Another frequent pncA variant in the active site."},

    # ── Second-line: Fluoroquinolones (FLQ) — gyrA ──
    {"gene": "gyrA",  "mutation": "D94G",     "drug": "LFX", "confidence": "assoc w resistance",
     "notes": "Most common FLQ mutation (~30-50% of FLQ-R). QRDR hotspot."},
    {"gene": "gyrA",  "mutation": "A90V",     "drug": "LFX", "confidence": "assoc w resistance",
     "notes": "Second most common FLQ mutation (~15-30%)."},

    # ── Second-line: Aminoglycosides — rrs (rRNA) + eis promoter ──
    {"gene": "rrs",   "mutation": "A1401G",   "drug": "AMK", "confidence": "assoc w resistance",
     "notes": "16S rRNA mutation. Most common AMK/KAN/CAP mutation (~80% of injectable-R)."},
    {"gene": "rrs",   "mutation": "C1402T",   "drug": "AMK", "confidence": "assoc w resistance",
     "notes": "Second rrs mutation for cross-resistance to aminoglycosides."},
    {"gene": "eis",   "mutation": "c.-14C>T", "drug": "KAN", "confidence": "assoc w resistance",
     "notes": "eis promoter mutation. Low-level KAN resistance (~30% of KAN-only-R)."},
]
# fmt: on


@dataclass
class TargetResult:
    """Summary for one mutation target."""
    gene: str
    mutation: str
    drug: str
    n_direct: int
    n_proximity: int
    n_total: int
    top_spacer: Optional[str]
    top_score: Optional[float]
    top_strategy: Optional[str]
    top_gc: Optional[float]
    top_pam: Optional[str]
    pam_desert: bool
    elapsed_sec: float
    notes: str


def parse_mutation(entry: dict) -> Mutation:
    """Parse a panel entry into a Mutation object.

    Handles three notations:
      - Amino acid:  S315T  → ref_aa=S, position=315, alt_aa=T
      - Promoter:    c.-15C>T → ref_aa=C, position=-15, alt_aa=T
      - rRNA:        A1401G → ref_aa=A, position=1401, alt_aa=G
    """
    gene = entry["gene"]
    raw = entry["mutation"]
    drug_str = entry["drug"]

    # Map drug string to Drug enum
    drug_map = {
        "RIF": Drug.RIFAMPICIN,
        "INH": Drug.ISONIAZID,
        "EMB": Drug.ETHAMBUTOL,
        "PZA": Drug.PYRAZINAMIDE,
        "LFX": Drug.LEVOFLOXACIN,
        "MFX": Drug.MOXIFLOXACIN,
        "AMK": Drug.AMIKACIN,
        "KAN": Drug.KANAMYCIN,
        "CAP": Drug.CAPREOMYCIN,
        "STR": Drug.STREPTOMYCIN,
        "BDQ": Drug.BEDAQUILINE,
        "LZD": Drug.LINEZOLID,
        "CFZ": Drug.CLOFAZIMINE,
        "DLM": Drug.DELAMANID,
        "ETH": Drug.ETHIONAMIDE,
    }
    drug = drug_map.get(drug_str)
    if drug is None:
        # Fallback: try direct enum lookup
        try:
            drug = Drug(drug_str)
        except ValueError:
            log.warning(f"Unknown drug '{drug_str}' for {gene}_{raw}, using RIFAMPICIN as placeholder")
            drug = Drug.RIFAMPICIN

    # ── Promoter notation: c.-15C>T ──
    if raw.startswith("c."):
        # Extract position and bases from c.-15C>T or c.-14C>T
        inner = raw[2:]  # "-15C>T"
        # Find where the ref base starts (first letter after digits/minus)
        i = 0
        while i < len(inner) and (inner[i].isdigit() or inner[i] == '-' or inner[i] == '+'):
            i += 1
        position = int(inner[:i])
        bases = inner[i:]  # "C>T"
        ref_base, alt_base = bases.split(">")
        return Mutation(
            gene=gene,
            position=position,
            ref_aa=ref_base.strip(),
            alt_aa=alt_base.strip(),
            nucleotide_change=raw,
            drug=drug,
            who_confidence=entry.get("confidence", ""),
        )

    # ── rRNA notation: A1401G (single letter + digits + single letter) ──
    # Detect: starts with a single nucleotide letter, ends with one
    if (len(raw) >= 3
            and raw[0] in "ACGT"
            and raw[-1] in "ACGT"
            and raw[1:-1].isdigit()):
        ref_base = raw[0]
        alt_base = raw[-1]
        position = int(raw[1:-1])
        return Mutation(
            gene=gene,
            position=position,
            ref_aa=ref_base,
            alt_aa=alt_base,
            nucleotide_change=raw,
            drug=drug,
            who_confidence=entry.get("confidence", ""),
        )

    # ── Standard amino acid: S315T ──
    ref_aa = raw[0]
    alt_aa = raw[-1]
    position = int(raw[1:-1])
    return Mutation(
        gene=gene,
        position=position,
        ref_aa=ref_aa,
        alt_aa=alt_aa,
        drug=drug,
        who_confidence=entry.get("confidence", ""),
    )


def run_panel(
    reference: str,
    gff: str,
    output_dir: str,
    mutations: Optional[list[dict]] = None,
) -> list[TargetResult]:
    """Run SABER on every mutation in the panel.

    Returns a list of TargetResult summaries.
    """
    if mutations is None:
        mutations = PANEL_MUTATIONS

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: list[TargetResult] = []

    for entry in mutations:
        gene = entry["gene"]
        raw_mut = entry["mutation"]
        label = f"{gene}_{raw_mut}".replace(".", "_").replace(">", "to").replace("-", "m")
        target_dir = out / label

        log.info(f"\n{'='*60}")
        log.info(f"  Designing: {gene} {raw_mut} ({entry['drug']})")
        log.info(f"  Notes: {entry.get('notes', '')}")
        log.info(f"{'='*60}")

        mutation = parse_mutation(entry)

        t0 = time.time()
        try:
            config = PipelineConfig(
                name=label,
                reference=ReferenceConfig(
                    genome_fasta=Path(reference),
                    gff_annotation=Path(gff) if gff else None,
                ),
                output_dir=str(target_dir),
            )
            pipeline = SABERPipeline(config=config)
            pipeline_results = pipeline.run([mutation])
            elapsed = time.time() - t0

            # Extract results for this target
            target_key = f"{gene}_{raw_mut}"
            scored = []
            for key, candidates in pipeline_results.items():
                scored.extend(candidates)

            n_direct = sum(1 for c in scored
                          if c.candidate.detection_strategy is None
                          or 'PROXIMITY' not in str(c.candidate.detection_strategy).upper())
            n_prox = len(scored) - n_direct

            if scored:
                # ScoredCandidate has: .candidate (CrRNACandidate), .heuristic, .rank
                # CrRNACandidate has: .spacer_seq, .pam_seq, .gc_content, .detection_strategy
                # HeuristicScore has: .composite
                def _score_of(s):
                    try:
                        return s.heuristic.composite
                    except Exception:
                        return 0.0

                top = max(scored, key=_score_of)
                cand = top.candidate  # CrRNACandidate
                top_spacer = cand.spacer_seq
                top_score = _score_of(top)
                top_strategy = str(cand.detection_strategy) if cand.detection_strategy else 'direct'
                top_gc = cand.gc_content
                top_pam = cand.pam_seq
            else:
                top_spacer = None
                top_score = None
                top_strategy = None
                top_gc = None
                top_pam = None

            result = TargetResult(
                gene=gene,
                mutation=raw_mut,
                drug=entry["drug"],
                n_direct=n_direct,
                n_proximity=n_prox,
                n_total=len(scored),
                top_spacer=top_spacer,
                top_score=top_score,
                top_strategy=top_strategy,
                top_gc=top_gc,
                top_pam=top_pam,
                pam_desert=(n_direct == 0 and n_prox > 0),
                elapsed_sec=round(elapsed, 2),
                notes=entry.get("notes", ""),
            )

        except Exception as e:
            elapsed = time.time() - t0
            log.error(f"FAILED {gene}_{raw_mut}: {e}", exc_info=True)
            result = TargetResult(
                gene=gene,
                mutation=raw_mut,
                drug=entry["drug"],
                n_direct=0,
                n_proximity=0,
                n_total=0,
                top_spacer=None,
                top_score=None,
                top_strategy=None,
                top_gc=None,
                top_pam=None,
                pam_desert=False,
                elapsed_sec=round(elapsed, 2),
                notes=f"ERROR: {e}",
            )

        results.append(result)
        log.info(f"  → {result.n_total} candidates "
                 f"({result.n_direct} direct, {result.n_proximity} proximity) "
                 f"in {result.elapsed_sec}s")

    # ── Write aggregated summary ──
    _write_summary(results, out)

    return results


def _write_summary(results: list[TargetResult], out: Path) -> None:
    """Write panel_summary.json and panel_summary.tsv."""

    # JSON
    summary_json = out / "panel_summary.json"
    with open(summary_json, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    log.info(f"Summary JSON: {summary_json}")

    # TSV for Excel / quick inspection
    summary_tsv = out / "panel_summary.tsv"
    cols = [
        "gene", "mutation", "drug", "n_total", "n_direct", "n_proximity",
        "pam_desert", "top_score", "top_gc", "top_pam", "top_strategy",
        "top_spacer", "elapsed_sec",
    ]
    with open(summary_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in results:
            rd = asdict(r)
            vals = []
            for c in cols:
                v = rd.get(c, "")
                if v is None:
                    v = ""
                elif isinstance(v, float):
                    v = f"{v:.3f}" if "score" in c or "gc" in c else f"{v:.1f}"
                elif isinstance(v, bool):
                    v = "YES" if v else ""
                vals.append(str(v))
            f.write("\t".join(vals) + "\n")
    log.info(f"Summary TSV: {summary_tsv}")

    # ── Print summary table to stdout ──
    print("\n" + "=" * 100)
    print("  MDR-TB 14-PLEX PANEL DESIGN SUMMARY")
    print("=" * 100)
    print(f"{'Gene':<8} {'Mutation':<12} {'Drug':<5} {'Total':>5} {'Direct':>6} "
          f"{'Prox':>5} {'Desert':>6} {'Score':>7} {'GC%':>5} {'PAM':<6} {'Strategy':<12}")
    print("-" * 100)

    total_candidates = 0
    total_direct = 0
    total_prox = 0
    n_deserts = 0
    n_failed = 0

    for r in results:
        total_candidates += r.n_total
        total_direct += r.n_direct
        total_prox += r.n_proximity
        if r.pam_desert:
            n_deserts += 1
        if r.n_total == 0:
            n_failed += 1

        score_str = f"{r.top_score:.3f}" if r.top_score is not None else "—"
        gc_str = f"{r.top_gc * 100:.0f}%" if r.top_gc is not None else "—"
        pam_str = r.top_pam or "—"
        strat_str = r.top_strategy or "—"
        desert_str = "🏜️" if r.pam_desert else ""

        print(f"{r.gene:<8} {r.mutation:<12} {r.drug:<5} {r.n_total:>5} {r.n_direct:>6} "
              f"{r.n_proximity:>5} {desert_str:>6} {score_str:>7} {gc_str:>5} {pam_str:<6} {strat_str:<12}")

    print("-" * 100)
    print(f"{'TOTAL':<26} {total_candidates:>5} {total_direct:>6} {total_prox:>5}")
    print(f"\nTargets: {len(results)} | With candidates: {len(results) - n_failed} | "
          f"PAM deserts: {n_deserts} | Failed: {n_failed}")
    print(f"Total candidates: {total_candidates} ({total_direct} direct, {total_prox} proximity)")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Design crRNAs for the complete MDR-TB 14-plex panel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reference", "-r", required=True,
        help="Path to H37Rv reference FASTA",
    )
    parser.add_argument(
        "--gff", "-g", required=True,
        help="Path to H37Rv GFF3 annotation",
    )
    parser.add_argument(
        "--output", "-o", default="results/mdr_14plex",
        help="Output directory (default: results/mdr_14plex)",
    )
    parser.add_argument(
        "--targets", "-t", nargs="*",
        help="Subset of targets to run (e.g., rpoB katG gyrA). Default: all.",
    )
    args = parser.parse_args()

    # Filter to requested targets if specified
    mutations = PANEL_MUTATIONS
    if args.targets:
        targets_lower = [t.lower() for t in args.targets]
        mutations = [m for m in mutations if m["gene"].lower() in targets_lower]
        if not mutations:
            log.error(f"No matching targets for: {args.targets}")
            log.info(f"Available genes: {sorted(set(m['gene'] for m in PANEL_MUTATIONS))}")
            sys.exit(1)

    log.info(f"Designing {len(mutations)} targets for MDR-TB panel")
    log.info(f"Reference: {args.reference}")
    log.info(f"Annotation: {args.gff}")
    log.info(f"Output: {args.output}")

    results = run_panel(
        reference=args.reference,
        gff=args.gff,
        output_dir=args.output,
        mutations=mutations,
    )

    # Exit code: 0 if all targets have candidates, 1 if any failed
    n_failed = sum(1 for r in results if r.n_total == 0)
    if n_failed:
        log.warning(f"{n_failed}/{len(results)} targets produced 0 candidates")
    sys.exit(1 if n_failed == len(results) else 0)


if __name__ == "__main__":
    main()

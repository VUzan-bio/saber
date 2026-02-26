"""Test synthetic mismatch enhancement on MDR-TB panel results."""
import json
import os
import sys
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

from saber.candidates.synthetic_mismatch import generate_enhanced_variants, EnhancementConfig

config = EnhancementConfig(cas_variant="enAsCas12a", allow_double_synthetic=True)

base = Path("results/mdr_14plex")
if not base.exists():
    print("No results found. Run design_core_panel.py first.")
    sys.exit(1)

for d in sorted(base.iterdir()):
    scored_path = d / "scored_candidates.json"
    if not scored_path.exists():
        continue
    with open(scored_path) as f:
        raw = json.load(f)
    # Handle both formats: dict keyed by target or flat list
    if isinstance(raw, dict):
        scored = []
        for key, val in raw.items():
            if isinstance(val, list):
                scored.extend(val)
            else:
                scored.append(val)
    else:
        scored = raw
    if not scored:
        continue

    print(f"\n{'='*60}")
    print(f"  {d.name} ({len(scored)} candidates)")
    print(f"{'='*60}")

    for sc in scored[:3]:
        cand = sc.get("candidate", sc)
        cid = cand["candidate_id"][:12]
        mm_pos = cand.get("mutation_position_in_spacer")

        if not mm_pos or mm_pos < 1:
            strategy = cand.get("detection_strategy", "unknown")
            print(f"  {cid}: proximity candidate — skipping (AS-RPA handles discrimination)")
            continue

        spacer = cand["spacer_seq"]
        mut_seq = spacer

        # Approximate WT by flipping the mutation base (transition)
        wt_seq = list(spacer)
        idx = mm_pos - 1
        flip = {"A": "G", "G": "A", "T": "C", "C": "T"}
        wt_seq[idx] = flip.get(wt_seq[idx], wt_seq[idx])
        wt_seq = "".join(wt_seq)

        report = generate_enhanced_variants(
            candidate_id=cand["candidate_id"],
            target_label=cand["target_label"],
            spacer_seq=mut_seq,
            wt_target_seq=wt_seq,
            mut_target_seq=mut_seq,
            natural_mm_position=mm_pos,
            config=config,
        )

        if report.best_variant:
            b = report.best_variant
            s = b.synthetic_mismatches[0]
            print(
                f"  {cid}: baseline={report.natural_discrimination_score:.0f}x "
                f"-> enhanced={report.best_discrimination_score:.0f}x "
                f"({report.improvement_factor:.1f}x improvement) "
                f"synthetic@pos{s.position} act_mut={b.predicted_activity_vs_mut:.2f}"
            )
        else:
            print(f"  {cid}: no viable enhancement")

print()

"""Pipeline configuration.

Loaded from YAML, validated with Pydantic. One config drives the entire run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ReferenceConfig(BaseModel):
    genome_fasta: Path
    genome_index: Optional[Path] = None             # pre-built Bowtie2 index
    human_index: Optional[Path] = None              # for off-target vs human
    gff_annotation: Optional[Path] = None


class CandidateConfig(BaseModel):
    spacer_lengths: list[int] = Field(default=[20, 21, 23])
    use_enascas12a: bool = True                     # also scan relaxed PAMs
    require_seed_mutation: bool = True               # mutation must be in pos 1-8
    gc_min: float = 0.40
    gc_max: float = 0.60
    homopolymer_max: int = 4
    mfe_threshold: float = -2.0


class ScoringConfig(BaseModel):
    use_heuristic: bool = True
    use_ml: bool = False
    ml_model_path: Optional[Path] = None            # Seq-deepCpf1 or JEPA checkpoint
    ml_model_name: str = "heuristic"


class MultiplexConfig(BaseModel):
    max_plex: int = 14
    optimizer: str = "greedy"                       # "greedy" | "simulated_annealing" | "cp"
    max_iterations: int = 10_000
    cross_reactivity_threshold: float = 0.3


class PrimerConfig(BaseModel):
    primer_length_min: int = 30
    primer_length_max: int = 35
    tm_min: float = 60.0
    tm_max: float = 65.0
    amplicon_min: int = 100
    amplicon_max: int = 200


class PipelineConfig(BaseModel):
    """Top-level config — one object drives the full pipeline."""
    name: str = "saber_run"
    output_dir: Path = Path("results")
    reference: ReferenceConfig
    candidates: CandidateConfig = CandidateConfig()
    scoring: ScoringConfig = ScoringConfig()
    multiplex: MultiplexConfig = MultiplexConfig()
    primers: PrimerConfig = PrimerConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

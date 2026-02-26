# SABER — Systematic Automated Biosensor Engineering for Resistance

Computational pipeline for designing, ranking, and optimizing CRISPR-Cas12a crRNA guides
targeting drug-resistance mutations in *Mycobacterium tuberculosis*.

## Overview

SABER automates the full crRNA design workflow for multiplexed electrochemical
CRISPR-Cas12a diagnostics:

<img width="2784" height="1536" alt="Gemini_Generated_Image_lc2mx3lc2mx3lc2m" src="https://github.com/user-attachments/assets/f9f58d2f-75c0-4f4a-af77-cc7264a0a995" />
<br>
<br>


1. **Target Definition** — Resolve genomic coordinates for WHO-catalogued resistance mutations
2. **Candidate Generation** — Scan for PAM-proximal spacers with mutation in seed region
3. **Off-Target Screening** — Align candidates against reference genomes
4. **Efficiency Scoring** — Heuristic rules → sequence-based ML → JEPA fine-tuned predictor
5. **Multiplex Optimization** — Select optimal N-plex panel under cross-reactivity constraints
6. **RPA Primer Co-Design** — Design amplification primers jointly with crRNA selection
7. **Experimental Tracking** — Log wet-lab results, close the active learning loop

## Installation

```bash
pip install -e ".[dev]"
```

### External dependencies

- [Bowtie2](https://github.com/BenLangmead/bowtie2) — off-target alignment
- [ViennaRNA](https://www.tbi.univie.ac.at/RNA/) — secondary structure prediction
- [Primer3](https://primer3.org/) — primer thermodynamics

## Quick start

```bash
# Design crRNAs for a single target
saber design --gene rpoB --mutation S531L --reference data/references/H37Rv.fasta

# Run full 14-plex panel design
saber panel --config configs/mdr_14plex.yaml

# Score candidates with JEPA (requires trained checkpoint)
saber score --candidates results/candidates.json --model checkpoints/jepa_cas12a.pt
```

## Project structure

```
saber/
├── core/           # Data models, configuration, constants
├── targets/        # Module 1: target definition from WHO catalogue
├── candidates/     # Module 2: crRNA candidate generation
├── offtarget/      # Module 3: off-target screening
├── scoring/        # Module 4: efficiency prediction (heuristic → JEPA)
├── multiplex/      # Module 5: multiplex panel optimization
├── primers/        # Module 6: RPA primer co-design
├── validation/     # Module 7: experimental result tracking
└── pipeline/       # Orchestration and CLI entry points
```

## License

MIT

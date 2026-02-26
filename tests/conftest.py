"""Shared test fixtures for SABER."""

import pytest

from saber.core.types import (
    CrRNACandidate,
    Drug,
    HeuristicScore,
    Mutation,
    OffTargetReport,
    PAMVariant,
    ScoredCandidate,
    Strand,
    Target,
)


@pytest.fixture
def rpob_s531l() -> Mutation:
    return Mutation(
        gene="rpoB",
        position=531,
        ref_aa="S",
        alt_aa="L",
        nucleotide_change="c.1349C>T",
        drug=Drug.RIFAMPICIN,
    )


@pytest.fixture
def rpob_target(rpob_s531l: Mutation) -> Target:
    return Target(
        mutation=rpob_s531l,
        genomic_pos=761155,
        ref_codon="TCG",
        alt_codon="TTG",
        flanking_seq="A" * 200 + "TTTGTCG" + "A" * 200,  # synthetic
        flanking_start=760955,
    )


@pytest.fixture
def sample_candidate() -> CrRNACandidate:
    return CrRNACandidate(
        candidate_id="test_abc123",
        target_label="rpoB_S531L",
        spacer_seq="GCGATCAAGGAGTTCTTCGG",
        pam_seq="TTTG",
        pam_variant=PAMVariant.TTTV,
        strand=Strand.PLUS,
        genomic_start=761140,
        genomic_end=761160,
        mutation_position_in_spacer=4,
        gc_content=0.55,
        homopolymer_max=2,
        mfe=-0.5,
    )


@pytest.fixture
def clean_offtarget(sample_candidate: CrRNACandidate) -> OffTargetReport:
    return OffTargetReport(
        candidate_id=sample_candidate.candidate_id,
        mtb_hits=[],
        human_hits=[],
        is_clean=True,
    )

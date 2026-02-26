"""Tests for core types and constants."""

import pytest

from saber.core.constants import pam_matches
from saber.core.types import (
    CrRNACandidate,
    DiscriminationScore,
    Drug,
    Mutation,
    OffTargetReport,
    PAMVariant,
    Strand,
)


class TestPAMMatching:
    def test_tttv_matches_tttg(self):
        assert pam_matches("TTTG", "TTTV") is True

    def test_tttv_matches_ttta(self):
        assert pam_matches("TTTA", "TTTV") is True

    def test_tttv_rejects_tttt(self):
        assert pam_matches("TTTT", "TTTV") is False

    def test_ttyn_matches_ttcn(self):
        assert pam_matches("TTCA", "TTYN") is True

    def test_case_insensitive(self):
        assert pam_matches("tttg", "TTTV") is True


class TestMutation:
    def test_label(self, rpob_s531l: Mutation):
        assert rpob_s531l.label == "rpoB_S531L"

    def test_drug(self, rpob_s531l: Mutation):
        assert rpob_s531l.drug == Drug.RIFAMPICIN


class TestCrRNACandidate:
    def test_in_seed_true(self, sample_candidate: CrRNACandidate):
        assert sample_candidate.in_seed is True

    def test_in_seed_false(self):
        c = CrRNACandidate(
            candidate_id="test",
            target_label="test",
            spacer_seq="GCGATCAAGGAGTTCTTCGG",
            pam_seq="TTTG",
            pam_variant=PAMVariant.TTTV,
            strand=Strand.PLUS,
            genomic_start=0,
            genomic_end=20,
            mutation_position_in_spacer=12,
            gc_content=0.55,
            homopolymer_max=2,
        )
        assert c.in_seed is False


class TestDiscriminationScore:
    def test_ratio(self):
        ds = DiscriminationScore(wt_activity=0.1, mut_activity=0.9)
        assert abs(ds.ratio - 9.0) < 1e-6

    def test_ratio_zero_wt(self):
        ds = DiscriminationScore(wt_activity=0.0, mut_activity=0.5)
        assert ds.ratio == float("inf")


class TestOffTargetReport:
    def test_clean_report(self, clean_offtarget: OffTargetReport):
        assert clean_offtarget.is_clean is True
        assert clean_offtarget.total_risky_hits == 0

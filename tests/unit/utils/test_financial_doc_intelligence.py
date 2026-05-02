"""Tests for openhands.app_server.utils.financial_doc_intelligence."""

import pytest

from openhands.app_server.utils.financial_doc_intelligence import (
    DocumentType,
    RiskFlag,
    classify_document,
    extract_entities,
    interpret,
)


# ---------------------------------------------------------------------------
# Sample documents (abbreviated but realistic)
# ---------------------------------------------------------------------------

EARNINGS_TEXT = """
Q3 2024 Earnings Release

For the third quarter of 2024, revenue of $4.2 billion exceeded analyst expectations.
Net income was $620 million, up 12% year-over-year.
Diluted earnings per share of $1.84, versus $1.65 in Q3 2023.
Full-year guidance raised to $16.5 billion in revenue.
Operating income increased 8% to $850 million.
"""

LOAN_TEXT = """
Facility Agreement dated 15 March 2024

The Borrower, Acme Manufacturing Ltd, and the Lender, First National Bank PLC,
hereby agree to a term loan with a principal amount of £50 million.
The interest rate shall be 5.25% per annum above SONIA.
The maturity date is 15 March 2029.
Financial covenants: net debt to EBITDA ratio shall not exceed 3.5x at any test date.
Financial covenants: interest cover ratio shall be no less than 2.0x.
An event of default shall occur if any representation is materially incorrect.
"""

AUDIT_TEXT = """
Independent Auditor's Report

We have audited the financial statements of Northgate Holdings PLC for the year
ended 31 December 2023.

In our opinion, the financial statements give a true and fair view.

Key audit matter: Revenue recognition under IFRS 15.
Key audit matter: Valuation of goodwill arising from the 2022 acquisition.

Going concern: The directors have identified a material uncertainty that may cast
substantial doubt on the company's ability to continue as a going concern.
The company is in advanced discussions with lenders regarding covenant waivers.

Signed by Deloitte LLP on 28 February 2024.
"""

REGULATORY_TEXT = """
Annual Report filed with the FCA on 30 April 2024
Form 10-K submission for the year ended December 2023.
Entity name: Global Finance Corp
Period ended 31 December 2023.
This disclosure satisfies MIFID II transparency obligations.
"""

PROSPECTUS_TEXT = """
Prospectus — Initial Public Offering

Issuer: TechVenture UK Ltd is offering 50 million ordinary shares.
Use of proceeds: funds will be used to expand the cloud infrastructure platform.
Aggregate offering size of £200 million.
Lead book-runner: Goldman Sachs International
Risk factors: investment in the company's securities involves significant risk.
"""

RISKY_EARNINGS_TEXT = """
Annual Results 2023

Revenue decreased by 18% compared to the prior year.
The company has been notified of a material weakness in its internal controls
over financial reporting. Prior period adjustment required for revenue restatement.
A class action lawsuit has been filed alleging misleading disclosures.
Net debt exceeds covenant limits; lenders have granted a temporary waiver of the
leverage ratio covenant.
The Board has identified going concern risk and the auditors have issued a
qualified opinion on the financial statements.
"""


# ---------------------------------------------------------------------------
# classify_document
# ---------------------------------------------------------------------------

class TestClassifyDocument:
    def test_classifies_earnings_report(self):
        result = classify_document(EARNINGS_TEXT)
        assert result.doc_type == DocumentType.EARNINGS_REPORT
        assert result.confidence > 0.5

    def test_classifies_loan_agreement(self):
        result = classify_document(LOAN_TEXT)
        assert result.doc_type == DocumentType.LOAN_AGREEMENT
        assert result.confidence > 0.5

    def test_classifies_audit_report(self):
        result = classify_document(AUDIT_TEXT)
        assert result.doc_type == DocumentType.AUDIT_REPORT
        assert result.confidence > 0.5

    def test_classifies_regulatory_filing(self):
        result = classify_document(REGULATORY_TEXT)
        assert result.doc_type == DocumentType.REGULATORY_FILING
        assert result.confidence > 0.5

    def test_classifies_prospectus(self):
        result = classify_document(PROSPECTUS_TEXT)
        assert result.doc_type == DocumentType.PROSPECTUS
        assert result.confidence > 0.5

    def test_unknown_returns_other(self):
        result = classify_document("Hello world, this is a generic text with no financial content.")
        assert result.doc_type == DocumentType.OTHER
        assert result.confidence == 0.0

    def test_matched_signals_populated(self):
        result = classify_document(EARNINGS_TEXT)
        assert len(result.matched_signals) > 0

    def test_confidence_bounded(self):
        result = classify_document(EARNINGS_TEXT)
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------

class TestExtractEntities:
    def test_earnings_revenue_extracted(self):
        result = extract_entities(EARNINGS_TEXT, DocumentType.EARNINGS_REPORT)
        assert result.doc_type == DocumentType.EARNINGS_REPORT
        assert result.entities.revenue is not None
        assert "4.2" in result.entities.revenue or "billion" in result.entities.revenue.lower()

    def test_earnings_eps_extracted(self):
        result = extract_entities(EARNINGS_TEXT, DocumentType.EARNINGS_REPORT)
        assert result.entities.eps is not None
        assert "1.84" in result.entities.eps

    def test_loan_principal_extracted(self):
        result = extract_entities(LOAN_TEXT, DocumentType.LOAN_AGREEMENT)
        assert result.entities.principal is not None
        assert "50" in result.entities.principal

    def test_loan_covenants_extracted(self):
        result = extract_entities(LOAN_TEXT, DocumentType.LOAN_AGREEMENT)
        assert len(result.entities.covenants) > 0

    def test_audit_opinion_extracted(self):
        result = extract_entities(AUDIT_TEXT, DocumentType.AUDIT_REPORT)
        assert result.entities.audit_opinion is not None

    def test_audit_key_matters_extracted(self):
        result = extract_entities(AUDIT_TEXT, DocumentType.AUDIT_REPORT)
        assert len(result.entities.key_audit_matters) > 0

    def test_regulatory_regulator_extracted(self):
        result = extract_entities(REGULATORY_TEXT, DocumentType.REGULATORY_FILING)
        assert result.entities.regulator is not None
        assert "FCA" in result.entities.regulator

    def test_prospectus_offering_type_extracted(self):
        result = extract_entities(PROSPECTUS_TEXT, DocumentType.PROSPECTUS)
        assert result.entities.offering_type is not None

    def test_other_doc_type_returns_empty(self):
        result = extract_entities("some text", DocumentType.OTHER)
        assert result.entities == {}
        assert result.extraction_confidence == 0.0

    def test_confidence_bounded(self):
        result = extract_entities(EARNINGS_TEXT, DocumentType.EARNINGS_REPORT)
        assert 0.0 <= result.extraction_confidence <= 1.0


# ---------------------------------------------------------------------------
# Risk flag detection (via interpret)
# ---------------------------------------------------------------------------

class TestRiskFlagDetection:
    def test_going_concern_detected(self):
        result = interpret(AUDIT_TEXT)
        assert RiskFlag.GOING_CONCERN in result.risk_flags

    def test_covenant_breach_detected(self):
        result = interpret(RISKY_EARNINGS_TEXT)
        assert RiskFlag.COVENANT_BREACH in result.risk_flags

    def test_material_weakness_detected(self):
        result = interpret(RISKY_EARNINGS_TEXT)
        assert RiskFlag.MATERIAL_WEAKNESS in result.risk_flags

    def test_restatement_detected(self):
        result = interpret(RISKY_EARNINGS_TEXT)
        assert RiskFlag.RESTATEMENT in result.risk_flags

    def test_litigation_risk_detected(self):
        result = interpret(RISKY_EARNINGS_TEXT)
        assert RiskFlag.LITIGATION_RISK in result.risk_flags

    def test_revenue_decline_detected(self):
        result = interpret(RISKY_EARNINGS_TEXT)
        assert RiskFlag.REVENUE_DECLINE in result.risk_flags

    def test_no_false_positive_on_clean_doc(self):
        result = interpret(EARNINGS_TEXT)
        assert RiskFlag.GOING_CONCERN not in result.risk_flags
        assert RiskFlag.MATERIAL_WEAKNESS not in result.risk_flags
        assert RiskFlag.RESTATEMENT not in result.risk_flags

    def test_risk_explanations_populated(self):
        result = interpret(AUDIT_TEXT)
        for flag in result.risk_flags:
            assert flag in result.risk_explanation
            assert len(result.risk_explanation[flag]) > 10


# ---------------------------------------------------------------------------
# interpret — full pipeline
# ---------------------------------------------------------------------------

class TestInterpret:
    def test_returns_correct_doc_type(self):
        result = interpret(EARNINGS_TEXT)
        assert result.doc_type == DocumentType.EARNINGS_REPORT

    def test_summary_non_empty(self):
        result = interpret(EARNINGS_TEXT)
        assert len(result.summary) > 20

    def test_verdict_elevated_for_high_risk(self):
        result = interpret(RISKY_EARNINGS_TEXT)
        assert "ELEVATED" in result.plain_english_verdict or "MODERATE" in result.plain_english_verdict

    def test_verdict_routine_for_clean_doc(self):
        result = interpret(EARNINGS_TEXT)
        assert "routine" in result.plain_english_verdict.lower() or "standard" in result.plain_english_verdict.lower()

    def test_key_figures_extracted(self):
        result = interpret(EARNINGS_TEXT)
        assert len(result.key_figures) > 0

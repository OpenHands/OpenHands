"""Financial document classification, entity extraction, and interpretability.

Provides fully self-contained heuristic analysis of financial documents —
no external APIs or ML models required.  The three public entry-points are:

    classify_document(text)  -> DocumentClassification
    extract_entities(text, doc_type)  -> ExtractedEntities
    interpret(text)  -> InterpretabilityOutput

Typical usage::

    result = interpret(raw_text)
    print(result.summary)
    print(result.risk_flags)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Public enums
# ---------------------------------------------------------------------------


class DocumentType(str, Enum):
    EARNINGS_REPORT = "earnings_report"
    REGULATORY_FILING = "regulatory_filing"
    LOAN_AGREEMENT = "loan_agreement"
    AUDIT_REPORT = "audit_report"
    PROSPECTUS = "prospectus"
    OTHER = "other"


class RiskFlag(str, Enum):
    GOING_CONCERN = "going_concern"
    MATERIAL_WEAKNESS = "material_weakness"
    REVENUE_DECLINE = "revenue_decline"
    HIGH_LEVERAGE = "high_leverage"
    LITIGATION_RISK = "litigation_risk"
    RESTATEMENT = "restatement"
    COVENANT_BREACH = "covenant_breach"
    AUDITOR_QUALIFICATION = "auditor_qualification"


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


@dataclass
class DocumentClassification:
    doc_type: DocumentType
    confidence: float  # 0.0 – 1.0
    matched_signals: list[str] = field(default_factory=list)


@dataclass
class EarningsEntities:
    revenue: str | None = None
    net_income: str | None = None
    eps: str | None = None
    guidance: str | None = None
    period: str | None = None


@dataclass
class LoanEntities:
    principal: str | None = None
    interest_rate: str | None = None
    maturity_date: str | None = None
    borrower: str | None = None
    lender: str | None = None
    covenants: list[str] = field(default_factory=list)


@dataclass
class RegulatoryEntities:
    filing_type: str | None = None
    entity_name: str | None = None
    filing_date: str | None = None
    regulator: str | None = None
    period_covered: str | None = None


@dataclass
class AuditEntities:
    auditor: str | None = None
    audit_opinion: str | None = None
    period: str | None = None
    material_weaknesses: list[str] = field(default_factory=list)
    key_audit_matters: list[str] = field(default_factory=list)


@dataclass
class ProspectusEntities:
    issuer: str | None = None
    offering_type: str | None = None
    offer_size: str | None = None
    use_of_proceeds: str | None = None
    lead_underwriter: str | None = None


@dataclass
class ExtractedEntities:
    doc_type: DocumentType
    entities: EarningsEntities | LoanEntities | RegulatoryEntities | AuditEntities | ProspectusEntities | dict[str, Any]
    extraction_confidence: float


@dataclass
class InterpretabilityOutput:
    doc_type: DocumentType
    summary: str
    risk_flags: list[RiskFlag]
    risk_explanation: dict[RiskFlag, str]
    key_figures: dict[str, str]
    plain_english_verdict: str


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_CLASSIFICATION_SIGNALS: dict[DocumentType, list[str]] = {
    DocumentType.EARNINGS_REPORT: [
        r"\bearnings per share\b",
        r"\beps\b",
        r"\brevenue\b.{0,40}\bquarter\b",
        r"\bnet income\b",
        r"\boperating income\b",
        r"\bguidance\b",
        r"\bfull.?year outlook\b",
        r"\bq[1-4]\s+\d{4}\b",
        r"\bearnings release\b",
    ],
    DocumentType.REGULATORY_FILING: [
        r"\bform\s+10-[kq]\b",
        r"\bsec filing\b",
        r"\bannual report\b.{0,20}\bcommission\b",
        r"\bfca\b",
        r"\bregulatory\b.{0,30}\bsubmission\b",
        r"\bprospectus\b.{0,20}\bregulat\b",
        r"\bdisclosure\b.{0,20}\bobligation\b",
        r"\bmifid\b",
        r"\bbasel\b",
    ],
    DocumentType.LOAN_AGREEMENT: [
        r"\bloan agreement\b",
        r"\bcredit agreement\b",
        r"\bterm loan\b",
        r"\bfacility agreement\b",
        r"\bborrower\b.{0,60}\blender\b",
        r"\bprincipal amount\b",
        r"\binterest rate\b.{0,40}\bper annum\b",
        r"\bmaturity date\b",
        r"\bfinancial covenants?\b",
        r"\bevent of default\b",
    ],
    DocumentType.AUDIT_REPORT: [
        r"\bindependent auditor",
        r"\bwe have audited\b",
        r"\baudit opinion\b",
        r"\bin our opinion\b.{0,60}\bfinancial statements\b",
        r"\bmaterial weakness\b",
        r"\bkey audit matter",
        r"\bgoing concern\b",
        r"\bqualified opinion\b",
        r"\badverse opinion\b",
    ],
    DocumentType.PROSPECTUS: [
        r"\bprospectus\b",
        r"\binitial public offering\b",
        r"\bipo\b",
        r"\buse of proceeds\b",
        r"\bunderwriter\b",
        r"\boffering price\b",
        r"\brisk factors\b.{0,40}\binvestment\b",
        r"\bsecurities\b.{0,30}\boffer\b",
    ],
}


def classify_document(text: str) -> DocumentClassification:
    """Classify a financial document by type using keyword heuristics.

    Args:
        text: Raw document text.

    Returns:
        DocumentClassification with doc_type, confidence, and matched signals.
    """
    lower = text.lower()
    scores: dict[DocumentType, list[str]] = {dt: [] for dt in _CLASSIFICATION_SIGNALS}

    for doc_type, patterns in _CLASSIFICATION_SIGNALS.items():
        for pattern in patterns:
            if re.search(pattern, lower):
                scores[doc_type].append(pattern)

    best_type = max(scores, key=lambda dt: len(scores[dt]))
    best_count = len(scores[best_type])

    if best_count == 0:
        return DocumentClassification(
            doc_type=DocumentType.OTHER,
            confidence=0.0,
            matched_signals=[],
        )

    total_signals = len(_CLASSIFICATION_SIGNALS[best_type])
    confidence = min(0.5 + (best_count / total_signals) * 0.5, 0.99)

    return DocumentClassification(
        doc_type=best_type,
        confidence=round(confidence, 2),
        matched_signals=scores[best_type],
    )


# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------

def _first_match(pattern: str, text: str, group: int = 1) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(group).strip() if m else None


def _all_matches(pattern: str, text: str, group: int = 1) -> list[str]:
    return [m.group(group).strip() for m in re.finditer(pattern, text, re.IGNORECASE)]


def _extract_earnings(text: str) -> EarningsEntities:
    return EarningsEntities(
        revenue=_first_match(
            r"(?:total\s+)?revenue[s]?\s+(?:of\s+|was\s+|were\s+)?([\$£€][\d,\.]+\s*(?:billion|million|bn|m)?)",
            text,
        ),
        net_income=_first_match(
            r"net\s+income\s+(?:of\s+|was\s+)?([\$£€][\d,\.]+\s*(?:billion|million|bn|m)?)",
            text,
        ),
        eps=_first_match(
            r"(?:diluted\s+)?(?:earnings|loss)\s+per\s+share\s+(?:of\s+|was\s+)?([\$£€][\d,\.]+)",
            text,
        ),
        guidance=_first_match(
            r"(?:full.?year|annual)\s+(?:revenue\s+)?guidance\s+(?:of\s+|raised?\s+to\s+)?(.{10,80}?)(?:\.|,|\n)",
            text,
        ),
        period=_first_match(
            r"(?:for\s+the\s+)?(?:(?:first|second|third|fourth|[1-4](?:st|nd|rd|th))\s+quarter|q[1-4])\s+(?:of\s+)?(\d{4})",
            text,
        ),
    )


def _extract_loan(text: str) -> LoanEntities:
    return LoanEntities(
        principal=_first_match(
            r"principal\s+(?:amount\s+)?(?:of\s+)?([\$£€][\d,\.]+\s*(?:billion|million|bn|m)?)",
            text,
        ),
        interest_rate=_first_match(
            r"interest\s+rate\s+(?:of\s+)?([\d\.]+\s*%(?:\s+per\s+annum)?)",
            text,
        ),
        maturity_date=_first_match(
            r"(?:maturity|repayment)\s+date\s+(?:of\s+|is\s+)?(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})",
            text,
        ),
        borrower=_first_match(
            r"(?:the\s+)?borrower[,\s]+(?:being\s+)?([A-Z][A-Za-z\s&,\.]+?)(?:\s*[,\(]|\s+agrees|\s+shall)",
            text,
        ),
        lender=_first_match(
            r"(?:the\s+)?lender[,\s]+(?:being\s+)?([A-Z][A-Za-z\s&,\.]+?)(?:\s*[,\(]|\s+agrees|\s+shall)",
            text,
        ),
        covenants=_all_matches(
            r"(?:financial\s+)?covenant[s]?[:\s]+([^\.]{10,120})\.",
            text,
        )[:3],
    )


def _extract_regulatory(text: str) -> RegulatoryEntities:
    return RegulatoryEntities(
        filing_type=_first_match(r"\b(form\s+10-[kq]|annual\s+report|mifid\s+ii?\s+report|fca\s+\w+\s+report)\b", text),
        entity_name=_first_match(r"(?:filed\s+by|registrant|entity\s+name)[:\s]+([A-Z][A-Za-z\s&,\.]+?)(?:\s*[,\n])", text),
        filing_date=_first_match(r"(?:filed|submission)\s+(?:on\s+|date[:\s]+)?(\d{1,2}[\s\/\-]\w+[\s\/\-]\d{4})", text),
        regulator=_first_match(r"\b(FCA|SEC|ESMA|FINRA|PRA|EBA|CFTC)\b", text),
        period_covered=_first_match(r"(?:period|year)\s+(?:ended?|ending)\s+(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})", text),
    )


def _extract_audit(text: str) -> AuditEntities:
    return AuditEntities(
        auditor=_first_match(r"(?:signed|issued)\s+by\s+([A-Z][A-Za-z\s&,\.]+?)\s*(?:LLP|Ltd|Limited|LLC|,|\n)", text),
        audit_opinion=_first_match(
            r"(unqualified opinion|qualified opinion|adverse opinion|disclaimer of opinion|unmodified opinion|true and fair view)",
            text,
        ),
        period=_first_match(r"(?:year|period)\s+ended?\s+(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})", text),
        material_weaknesses=_all_matches(
            r"material\s+weakness[es]?\s+(?:in|relating\s+to|identified)[:\s]+([^\.]{10,120})\.",
            text,
        )[:5],
        key_audit_matters=_all_matches(
            r"key\s+audit\s+matter[s]?[:\s—–]+([^\.]{10,120})\.",
            text,
        )[:5],
    )


def _extract_prospectus(text: str) -> ProspectusEntities:
    return ProspectusEntities(
        issuer=_first_match(r"(?:issuer|company)[:\s]+([A-Z][A-Za-z\s&,\.]+?)(?:\s*[,\(]|\s+is\s+offering|\s+intends)", text),
        offering_type=_first_match(r"\b(initial\s+public\s+offering|follow.?on\s+offering|rights\s+issue|bond\s+issuance|ipo)\b", text),
        offer_size=_first_match(r"(?:aggregate|total)\s+(?:offering|offer)\s+(?:size|amount)\s+(?:of\s+)?([\$£€][\d,\.]+\s*(?:billion|million|bn|m)?)", text),
        use_of_proceeds=_first_match(r"use\s+of\s+proceeds[:\s]+([^\.]{10,200})\.", text),
        lead_underwriter=_first_match(r"(?:lead|joint\s+lead)\s+(?:book.?runner|underwriter)[:\s]+([A-Z][A-Za-z\s&,\.]+?)(?:\s*[,\n])", text),
    )


def extract_entities(text: str, doc_type: DocumentType) -> ExtractedEntities:
    """Extract structured entities from a financial document.

    Args:
        text: Raw document text.
        doc_type: The document type (use classify_document() if unknown).

    Returns:
        ExtractedEntities with type-specific fields populated where found.
    """
    extractors = {
        DocumentType.EARNINGS_REPORT: _extract_earnings,
        DocumentType.LOAN_AGREEMENT: _extract_loan,
        DocumentType.REGULATORY_FILING: _extract_regulatory,
        DocumentType.AUDIT_REPORT: _extract_audit,
        DocumentType.PROSPECTUS: _extract_prospectus,
    }

    extractor = extractors.get(doc_type)
    if extractor is None:
        return ExtractedEntities(
            doc_type=DocumentType.OTHER,
            entities={},
            extraction_confidence=0.0,
        )

    entities = extractor(text)

    # Confidence: ratio of non-None fields extracted
    entity_dict = entities.__dict__
    filled = sum(1 for v in entity_dict.values() if v not in (None, [], ""))
    total = len(entity_dict)
    confidence = round(filled / total, 2) if total > 0 else 0.0

    return ExtractedEntities(
        doc_type=doc_type,
        entities=entities,
        extraction_confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Risk flag detection
# ---------------------------------------------------------------------------

_RISK_PATTERNS: dict[RiskFlag, list[str]] = {
    RiskFlag.GOING_CONCERN: [
        r"\bgoing.?concern\b",
        r"\bsubstantial\s+doubt\b.{0,40}\bcontinue\b",
        r"\bability\s+to\s+continue\s+as\s+a\s+going\s+concern\b",
    ],
    RiskFlag.MATERIAL_WEAKNESS: [
        r"\bmaterial\s+weakness\b",
        r"\bsignificant\s+deficiency\b",
        r"\binternal\s+control\s+deficiency\b",
    ],
    RiskFlag.REVENUE_DECLINE: [
        r"revenue[s]?\s+(?:decreased?|declined?|fell|dropped)\s+(?:by\s+)?(\d+)",
        r"(?:year.?over.?year|yoy)\s+(?:revenue\s+)?(?:decrease|decline)\s+of\s+(\d+)",
        r"net\s+(?:revenue|sales)\s+(?:down|lower)\s+(?:by\s+)?(\d+)",
    ],
    RiskFlag.HIGH_LEVERAGE: [
        r"\bhighly?\s+leveraged\b",
        r"\bdebt.?to.?(?:equity|ebitda)\s+ratio\b.{0,40}\b[5-9]\d*[\.x]\b",
        r"\bnet\s+debt\b.{0,60}\b(?:exceed|above|over)\b",
        r"\bleverage\s+ratio\b.{0,40}\bbreached?\b",
    ],
    RiskFlag.LITIGATION_RISK: [
        r"\bpending\s+litigation\b",
        r"\bmaterial\s+(?:legal\s+)?proceedings\b",
        r"\bclass\s+action\b",
        r"\bregulatory\s+(?:investigation|enforcement)\b",
        r"\bcease\s+and\s+desist\b",
    ],
    RiskFlag.RESTATEMENT: [
        r"\brestat(?:ed?|ement)\b",
        r"\bprior\s+period\s+(?:adjustment|correction)\b",
        r"\bmaterial\s+(?:error|misstatement)\b",
    ],
    RiskFlag.COVENANT_BREACH: [
        r"\bcovenant\s+(?:breach|violation|waiver)\b",
        r"\bwaivers?\s+(?:of|from)\s+(?:the\s+)?lender\b",
        r"\blenders?\s+(?:have\s+)?granted\s+(?:a\s+)?(?:temporary\s+)?waiver\b",
        r"\bnon.?compliance\b.{0,40}\bcovenant\b",
        r"\bcross.?default\b",
    ],
    RiskFlag.AUDITOR_QUALIFICATION: [
        r"\bqualified\s+opinion\b",
        r"\badverse\s+opinion\b",
        r"\bdisclaimer\s+of\s+opinion\b",
        r"\bexcept\s+for\b.{0,40}\bopinion\b",
    ],
}

_RISK_EXPLANATIONS: dict[RiskFlag, str] = {
    RiskFlag.GOING_CONCERN: "Auditors have expressed doubt about the company's ability to continue operating as a going concern.",
    RiskFlag.MATERIAL_WEAKNESS: "A significant deficiency in internal controls has been identified, increasing the risk of financial misstatement.",
    RiskFlag.REVENUE_DECLINE: "Revenue has decreased year-over-year, indicating potential deterioration in business performance.",
    RiskFlag.HIGH_LEVERAGE: "The company carries a high debt load relative to earnings or equity, increasing financial risk.",
    RiskFlag.LITIGATION_RISK: "Material legal proceedings or regulatory investigations are pending that could result in significant financial liability.",
    RiskFlag.RESTATEMENT: "Previously filed financial statements have been or are being restated, indicating past reporting errors.",
    RiskFlag.COVENANT_BREACH: "Financial covenants have been breached or waived, signalling potential liquidity or compliance issues.",
    RiskFlag.AUDITOR_QUALIFICATION: "The auditor has issued a qualified, adverse, or disclaimer opinion rather than a clean unqualified opinion.",
}


def _detect_risk_flags(text: str) -> dict[RiskFlag, str]:
    """Returns a dict of detected RiskFlag -> explanation."""
    lower = text.lower()
    detected: dict[RiskFlag, str] = {}
    for flag, patterns in _RISK_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lower):
                detected[flag] = _RISK_EXPLANATIONS[flag]
                break
    return detected


# ---------------------------------------------------------------------------
# Key figures extraction (generic across doc types)
# ---------------------------------------------------------------------------

_FIGURE_PATTERNS: dict[str, str] = {
    "revenue": r"(?:total\s+)?revenue[s]?\s+(?:of\s+|was\s+|were\s+)?([\$£€][\d,\.]+\s*(?:billion|million|bn|m)?)",
    "net_income": r"net\s+(?:income|profit|loss)\s+(?:of\s+|was\s+|were\s+)?([\$£€][\d,\.]+\s*(?:billion|million|bn|m)?)",
    "eps": r"(?:diluted\s+)?(?:earnings|loss)\s+per\s+share\s+(?:of\s+|was\s+)?([\$£€]?[\d,\.]+)",
    "total_assets": r"total\s+assets\s+(?:of\s+|were\s+|amounted\s+to\s+)?([\$£€][\d,\.]+\s*(?:billion|million|bn|m)?)",
    "total_debt": r"total\s+(?:debt|borrowings?)\s+(?:of\s+|were\s+|amounted\s+to\s+)?([\$£€][\d,\.]+\s*(?:billion|million|bn|m)?)",
    "ebitda": r"\bebitda\b\s+(?:of\s+|was\s+|were\s+)?([\$£€][\d,\.]+\s*(?:billion|million|bn|m)?)",
}


def _extract_key_figures(text: str) -> dict[str, str]:
    figures: dict[str, str] = {}
    for label, pattern in _FIGURE_PATTERNS.items():
        val = _first_match(pattern, text)
        if val:
            figures[label] = val
    return figures


# ---------------------------------------------------------------------------
# Plain-English verdict
# ---------------------------------------------------------------------------

def _build_verdict(doc_type: DocumentType, risk_flags: list[RiskFlag]) -> str:
    if not risk_flags:
        return (
            f"This {doc_type.value.replace('_', ' ')} appears routine with no major risk indicators detected. "
            "Standard review procedures should be sufficient."
        )

    severity_map = {
        RiskFlag.GOING_CONCERN: 3,
        RiskFlag.AUDITOR_QUALIFICATION: 3,
        RiskFlag.MATERIAL_WEAKNESS: 2,
        RiskFlag.RESTATEMENT: 2,
        RiskFlag.COVENANT_BREACH: 2,
        RiskFlag.HIGH_LEVERAGE: 1,
        RiskFlag.LITIGATION_RISK: 1,
        RiskFlag.REVENUE_DECLINE: 1,
    }
    max_severity = max(severity_map.get(f, 0) for f in risk_flags)

    flag_names = ", ".join(f.value.replace("_", " ") for f in risk_flags)

    if max_severity >= 3:
        return (
            f"ELEVATED RISK — This {doc_type.value.replace('_', ' ')} contains critical risk signals: {flag_names}. "
            "Immediate escalation to senior compliance review is recommended before taking any action."
        )
    if max_severity == 2:
        return (
            f"MODERATE RISK — This {doc_type.value.replace('_', ' ')} contains notable risk signals: {flag_names}. "
            "Enhanced due diligence and secondary review are recommended."
        )
    return (
        f"LOW-MODERATE RISK — This {doc_type.value.replace('_', ' ')} shows minor risk signals: {flag_names}. "
        "Proceed with standard enhanced monitoring."
    )


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def interpret(text: str) -> InterpretabilityOutput:
    """Classify, extract entities, detect risk flags, and produce a plain-English summary.

    Args:
        text: Raw financial document text.

    Returns:
        InterpretabilityOutput with summary, risk_flags, risk_explanation,
        key_figures, and a plain_english_verdict.
    """
    classification = classify_document(text)
    doc_type = classification.doc_type

    risk_map = _detect_risk_flags(text)
    risk_flags = list(risk_map.keys())
    key_figures = _extract_key_figures(text)

    doc_type_label = doc_type.value.replace("_", " ").title()
    confidence_pct = int(classification.confidence * 100)
    flag_summary = (
        f" Risk signals detected: {', '.join(f.value.replace('_', ' ') for f in risk_flags)}."
        if risk_flags
        else " No major risk signals detected."
    )

    summary = (
        f"Document classified as {doc_type_label} (confidence: {confidence_pct}%).{flag_summary}"
    )
    if key_figures:
        figures_text = "; ".join(f"{k.replace('_',' ')}: {v}" for k, v in key_figures.items())
        summary += f" Key figures: {figures_text}."

    return InterpretabilityOutput(
        doc_type=doc_type,
        summary=summary,
        risk_flags=risk_flags,
        risk_explanation=risk_map,
        key_figures=key_figures,
        plain_english_verdict=_build_verdict(doc_type, risk_flags),
    )

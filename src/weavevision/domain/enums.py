"""Stable domain enumerations used across WeaveVision."""

from enum import StrEnum


class Decision(StrEnum):
    """Analysis decision that never implies calibrated probability."""

    NORMAL = "NORMAL"
    ANOMALY = "ANOMALY"
    REVIEW = "REVIEW"
    ABSTAIN = "ABSTAIN"


class QualityGateStatus(StrEnum):
    """Input quality gate outcome."""

    PASS = "PASS"
    REVIEW = "REVIEW"
    ABSTAIN = "ABSTAIN"


class ReviewPriority(StrEnum):
    """Operational review priority, not a physical defect severity class."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    ABSTAIN = "ABSTAIN"


class ModelStatus(StrEnum):
    """Model registry lifecycle state."""

    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    ACTIVE_BENCHMARK = "ACTIVE_BENCHMARK"
    ACTIVE_COMPANY_PILOT = "ACTIVE_COMPANY_PILOT"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class ExperimentStatus(StrEnum):
    """Evidence status for experiments and acceptance gates."""

    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    PASS_WITH_RESTRICTIONS = "PASS_WITH_RESTRICTIONS"


class FeedbackVerdict(StrEnum):
    """Quality expert feedback verdict."""

    CONFIRMED_NORMAL = "CONFIRMED_NORMAL"
    CONFIRMED_ANOMALY = "CONFIRMED_ANOMALY"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    UNSURE = "UNSURE"


class DatasetVerificationStatus(StrEnum):
    """Dataset governance verification state."""

    VERIFIED = "VERIFIED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"

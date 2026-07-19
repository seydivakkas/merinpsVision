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


# ---------------------------------------------------------------------------
# Drift lifecycle enumerations (M1)
# ---------------------------------------------------------------------------


class DriftPattern(StrEnum):
    """Observed drift pattern classification used in incident triage."""

    STABLE = "STABLE"
    SUDDEN = "SUDDEN"
    GRADUAL = "GRADUAL"
    SEMANTIC = "SEMANTIC"
    TECHNICAL = "TECHNICAL"


class TrendStatus(StrEnum):
    """Combined EWMA + CUSUM alert state for a single monitoring window."""

    STABLE = "STABLE"
    EWMA_ALERT = "EWMA_ALERT"
    CUSUM_ALERT = "CUSUM_ALERT"
    BOTH_ALERT = "BOTH_ALERT"


class IncidentPriority(StrEnum):
    """Operational incident priority derived from confirming signal count."""

    INFO = "INFO"
    P2_REVIEW = "P2_REVIEW"
    P1_INCIDENT = "P1_INCIDENT"
    P0_BLOCKED = "P0_BLOCKED"


class RetrainingStrategy(StrEnum):
    """Retraining strategy chosen during incident triage."""

    NONE = "NONE"
    FINE_TUNE = "FINE_TUNE"
    FULL_RETRAIN = "FULL_RETRAIN"
    CONTINUAL = "CONTINUAL"


class CanaryStatus(StrEnum):
    """Canary evaluation lifecycle state."""

    NOT_RUN = "NOT_RUN"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"


class RollbackReason(StrEnum):
    """Reason that triggered a model rollback event."""

    HASH_MISMATCH = "HASH_MISMATCH"
    RECALL_DROP = "RECALL_DROP"
    FP_SPIKE = "FP_SPIKE"
    LATENCY = "LATENCY"
    DRIFT_WORSENING = "DRIFT_WORSENING"
    SAFETY_ALARM = "SAFETY_ALARM"

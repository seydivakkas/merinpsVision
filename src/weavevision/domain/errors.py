"""Typed domain errors with stable public error codes."""


class WeaveVisionError(Exception):
    """Base exception carrying a stable machine-readable error code."""

    code = "WV_UNKNOWN"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigError(WeaveVisionError):
    """Raised when application or experiment configuration is invalid."""

    code = "WV_CONFIG_INVALID"


class DatasetNotFoundError(WeaveVisionError):
    """Raised when a configured dataset is unavailable."""

    code = "WV_DATASET_NOT_FOUND"


class DatasetLicenseBlockedError(WeaveVisionError):
    """Raised when dataset access requires unresolved license acceptance."""

    code = "WV_DATASET_LICENSE_BLOCKED"


class DatasetStructureError(WeaveVisionError):
    """Raised when dataset layout or content violates its contract."""

    code = "WV_DATASET_STRUCTURE_INVALID"


class DataLeakageError(WeaveVisionError):
    """Raised when source identity or duplicate leakage crosses splits."""

    code = "WV_DATA_LEAKAGE_DETECTED"


class ModelNotReadyError(WeaveVisionError):
    """Raised when no eligible model can serve an analysis request."""

    code = "WV_MODEL_NOT_READY"


class ModelHashMismatchError(WeaveVisionError):
    """Raised when a model artifact fails its integrity check."""

    code = "WV_MODEL_HASH_MISMATCH"


class ThresholdNotFoundError(WeaveVisionError):
    """Raised when production decision thresholds are unavailable."""

    code = "WV_THRESHOLD_NOT_FOUND"


class ImageValidationError(WeaveVisionError):
    """Raised when an image cannot be safely decoded."""

    code = "WV_IMAGE_INVALID"


class UnsupportedFormatError(WeaveVisionError):
    """Raised when an input format is unsupported."""

    code = "WV_UNSUPPORTED_FORMAT"


class InferenceError(WeaveVisionError):
    """Raised when model inference fails at the service boundary."""

    code = "WV_INFERENCE_FAILED"


class ReportError(WeaveVisionError):
    """Raised when a report artifact cannot be produced."""

    code = "WV_REPORT_FAILED"


class DatabaseError(WeaveVisionError):
    """Raised for persistence failures."""

    code = "WV_DATABASE_FAILED"

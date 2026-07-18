"""VisA adapter boundary; download and transformation remain separate operations."""

from weavevision.data.adapters.mvtec_ad import MVTecADAdapter


class VisAAdapter(MVTecADAdapter):
    """Verify a normalized VisA category in the common governed layout."""

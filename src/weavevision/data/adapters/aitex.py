"""AITEX adapter boundary enforcing parent-image identity before tiling."""

from weavevision.data.adapters.mvtec_ad import MVTecADAdapter


class AITEXAdapter(MVTecADAdapter):
    """Verify a normalized AITEX category while retaining source-image groups."""

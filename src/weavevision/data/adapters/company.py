"""Company pilot import contract validation."""

from __future__ import annotations

from weavevision.data.adapters.mvtec_ad import MVTecADAdapter


class CompanyDatasetAdapter(MVTecADAdapter):
    """Folder-based pilot adapter reusing strict image and mask verification.

    Company promotion remains separate from benchmark model promotion. The current adapter
    expects the governed MVTec-like train/test layout after an authorized import step.
    """

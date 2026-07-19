"""Unit tests for evaluation/psi.py (M2).

Senaryolar:
- Ayni dagilim -> PSI ~ 0
- Kaydirilmis dagilim -> PSI yuksek
- Bos dizi -> ValueError
- bins < 2 -> ValueError
- psi_severity sinir testleri
- psi_severity medium >= high -> ValueError
"""

from __future__ import annotations

import numpy as np
import pytest

from weavevision.evaluation.psi import population_stability_index, psi_severity


class TestPopulationStabilityIndex:
    """population_stability_index() icin testler."""

    def test_identical_distributions_near_zero(self) -> None:
        """Ayni dagilim -> PSI epsilon etrafinda cok dusuk olmali."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=0.85, scale=0.05, size=1000)
        psi = population_stability_index(data, data.copy())
        assert psi < 0.05, f"Ayni dagilim icin PSI cok dusuk beklenir, got {psi:.4f}"

    def test_shifted_distribution_high_psi(self) -> None:
        """Belirgin ort. kaymasi -> PSI > 0.25 (HIGH)."""
        rng = np.random.default_rng(99)
        reference = rng.normal(loc=0.85, scale=0.05, size=500)
        current = rng.normal(loc=0.60, scale=0.05, size=500)
        psi = population_stability_index(reference, current)
        assert psi > 0.25, f"Buyuk kayma icin PSI > 0.25 beklenir, got {psi:.4f}"

    def test_moderate_shift_non_negative(self) -> None:
        """Orta duzey kayma -> gecerli float donus."""
        rng = np.random.default_rng(7)
        reference = rng.normal(loc=0.85, scale=0.05, size=1000)
        current = rng.normal(loc=0.78, scale=0.05, size=1000)
        psi = population_stability_index(reference, current)
        assert psi >= 0.0

    def test_psi_non_negative(self) -> None:
        """PSI her zaman >= 0 olmali."""
        rng = np.random.default_rng(13)
        a = rng.uniform(0, 1, 200)
        b = rng.uniform(0, 1, 200)
        assert population_stability_index(a, b) >= 0.0

    def test_empty_reference_raises(self) -> None:
        """Bos referans -> ValueError."""
        with pytest.raises(ValueError, match="bos olamaz"):
            population_stability_index(np.array([]), np.array([0.85]))

    def test_empty_current_raises(self) -> None:
        """Bos current -> ValueError."""
        with pytest.raises(ValueError, match="bos olamaz"):
            population_stability_index(np.array([0.85]), np.array([]))

    def test_bins_less_than_2_raises(self) -> None:
        """bins < 2 -> ValueError."""
        with pytest.raises(ValueError, match="bins"):
            population_stability_index(np.array([0.85, 0.84]), np.array([0.83, 0.82]), bins=1)

    def test_different_sizes_allowed(self) -> None:
        """Referans ve current farkli boyutta olabilir."""
        rng = np.random.default_rng(0)
        reference = rng.normal(0.85, 0.05, 300)
        current = rng.normal(0.85, 0.05, 100)
        psi = population_stability_index(reference, current)
        assert psi >= 0.0

    def test_single_element_arrays(self) -> None:
        """Tekil eleman dizileri hata vermemeli (bins=2 ile)."""
        psi = population_stability_index(np.array([0.85]), np.array([0.84]), bins=2)
        assert psi >= 0.0

    def test_custom_bins(self) -> None:
        """bins parametresi gecerli degerlerle calismali."""
        rng = np.random.default_rng(42)
        a = rng.normal(0.85, 0.05, 200)
        b = rng.normal(0.80, 0.05, 200)
        psi5 = population_stability_index(a, b, bins=5)
        psi20 = population_stability_index(a, b, bins=20)
        assert isinstance(psi5, float)
        assert isinstance(psi20, float)


class TestPsiSeverity:
    """psi_severity() icin testler."""

    def test_low_severity(self) -> None:
        assert psi_severity(0.05) == "LOW"

    def test_medium_severity(self) -> None:
        assert psi_severity(0.15) == "MEDIUM"

    def test_high_severity(self) -> None:
        assert psi_severity(0.30) == "HIGH"

    def test_exact_medium_threshold_is_medium(self) -> None:
        """Esit deger MEDIUM'a girmeli (< high)."""
        assert psi_severity(0.10) == "MEDIUM"

    def test_exact_high_threshold_is_high(self) -> None:
        """Esit deger HIGH'a girmeli."""
        assert psi_severity(0.25) == "HIGH"

    def test_zero_psi_is_low(self) -> None:
        assert psi_severity(0.0) == "LOW"

    def test_custom_thresholds(self) -> None:
        """Parametre gecersiz kilma calismali."""
        assert psi_severity(0.05, medium_threshold=0.08, high_threshold=0.15) == "LOW"
        assert psi_severity(0.10, medium_threshold=0.08, high_threshold=0.15) == "MEDIUM"
        assert psi_severity(0.20, medium_threshold=0.08, high_threshold=0.15) == "HIGH"

    def test_medium_ge_high_raises(self) -> None:
        """medium_threshold >= high_threshold -> ValueError."""
        with pytest.raises(ValueError, match="medium_threshold"):
            psi_severity(0.10, medium_threshold=0.25, high_threshold=0.10)

    def test_medium_equal_high_raises(self) -> None:
        with pytest.raises(ValueError):
            psi_severity(0.10, medium_threshold=0.20, high_threshold=0.20)

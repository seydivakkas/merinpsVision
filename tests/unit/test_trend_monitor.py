"""Unit tests for evaluation/trend_monitor.py (M2).

Senaryolar:
- Stabil seri: hicbir uyari tetiklenmemeli
- Ani dusus: hem EWMA hem CUSUM uyari vermeli
- 3+ hafta kademeli dusus: CUSUM birikmeli
- baseline_std=0 -> ValueError
- values bos -> ValueError
- ewma_lambda sinir disi -> ValueError
- classify_sudden_drop sinir testleri
"""

from __future__ import annotations

import numpy as np
import pytest

from weavevision.evaluation.trend_monitor import (
    TrendPoint,
    classify_sudden_drop,
    monitor_downward_drift,
)


class TestMonitorDownwardDrift:
    """monitor_downward_drift() fonksiyonu icin kapsamli testler."""

    def test_stable_series_no_alerts(self) -> None:
        """Ortalama etrafinda rasgele salinimda uyari tetiklenmemeli."""
        rng = np.random.default_rng(42)
        values = rng.normal(loc=0.85, scale=0.005, size=20)
        points = monitor_downward_drift(
            values,
            baseline_mean=0.85,
            baseline_std=0.01,
            ewma_lambda=0.25,
            cusum_h_sigma=4.0,
        )
        assert len(points) == 20
        assert all(p.status == "STABLE" for p in points)
        assert all(not p.ewma_alert for p in points)
        assert all(not p.cusum_alert for p in points)

    def test_sudden_drop_triggers_both_alerts(self) -> None:
        """Ani buyuk dusus hem EWMA hem CUSUM uyarisi uretmeli."""
        # Ilk 10 deger stabil, son 5 deger dramatik dusus
        stable = np.full(10, 0.85)
        dropped = np.full(5, 0.60)
        values = np.concatenate([stable, dropped])
        points = monitor_downward_drift(
            values,
            baseline_mean=0.85,
            baseline_std=0.02,
            ewma_lambda=0.5,
            cusum_h_sigma=3.0,
        )
        last_points = points[-3:]
        assert any(p.cusum_alert for p in last_points), "CUSUM dususu tespit etmeli"
        assert any(p.ewma_alert for p in last_points), "EWMA dususu tespit etmeli"

    def test_gradual_decline_cusum_accumulates(self) -> None:
        """3 hafta kademeli dusus -- CUSUM birikmeli."""
        # Her hafta 0.5pp dusus
        weekly_means = [0.85, 0.845, 0.840, 0.835, 0.830, 0.825]
        values = np.array(weekly_means)
        points = monitor_downward_drift(
            values,
            baseline_mean=0.85,
            baseline_std=0.005,
            ewma_lambda=0.2,
            cusum_k_sigma=0.1,
            cusum_h_sigma=2.0,
        )
        cusum_values = [p.cusum_down for p in points]
        assert cusum_values[-1] > cusum_values[0], "CUSUM kademeli dususu biriktirmeli"

    def test_index_starts_at_1(self) -> None:
        """TrendPoint.index 1'den baslamali."""
        values = np.array([0.85, 0.84, 0.83])
        points = monitor_downward_drift(values, baseline_mean=0.85, baseline_std=0.01)
        assert [p.index for p in points] == [1, 2, 3]

    def test_returns_trend_point_dataclass(self) -> None:
        """Donus tipi TrendPoint frozen dataclass olmali."""
        values = np.array([0.85])
        points = monitor_downward_drift(values, baseline_mean=0.85, baseline_std=0.01)
        assert isinstance(points[0], TrendPoint)
        with pytest.raises((AttributeError, TypeError)):
            points[0].value = 0.99  # type: ignore[misc]  # frozen

    def test_baseline_std_zero_raises(self) -> None:
        """baseline_std=0 -> ValueError."""
        with pytest.raises(ValueError, match="baseline_std"):
            monitor_downward_drift(np.array([0.85]), baseline_mean=0.85, baseline_std=0.0)

    def test_empty_values_raises(self) -> None:
        """Bos dizi -> ValueError."""
        with pytest.raises(ValueError, match="bos olmayan"):
            monitor_downward_drift(np.array([]), baseline_mean=0.85, baseline_std=0.01)

    def test_2d_values_raises(self) -> None:
        """2-B dizi -> ValueError."""
        with pytest.raises(ValueError, match="tek boyutlu"):
            monitor_downward_drift(np.array([[0.85, 0.84]]), baseline_mean=0.85, baseline_std=0.01)

    def test_ewma_lambda_zero_raises(self) -> None:
        """ewma_lambda=0 -> ValueError (aralik disi)."""
        with pytest.raises(ValueError, match="ewma_lambda"):
            monitor_downward_drift(
                np.array([0.85]),
                baseline_mean=0.85,
                baseline_std=0.01,
                ewma_lambda=0.0,
            )

    def test_ewma_lambda_gt_1_raises(self) -> None:
        """ewma_lambda=1.1 -> ValueError."""
        with pytest.raises(ValueError, match="ewma_lambda"):
            monitor_downward_drift(
                np.array([0.85]),
                baseline_mean=0.85,
                baseline_std=0.01,
                ewma_lambda=1.1,
            )

    def test_ewma_lambda_1_is_valid(self) -> None:
        """ewma_lambda=1.0 gecerli (sinir degeri)."""
        values = np.array([0.85, 0.83])
        points = monitor_downward_drift(
            values, baseline_mean=0.85, baseline_std=0.01, ewma_lambda=1.0
        )
        assert len(points) == 2

    def test_cusum_never_negative(self) -> None:
        """CUSUM istatistigi asla negatif olmaz."""
        rng = np.random.default_rng(7)
        values = rng.normal(loc=0.90, scale=0.01, size=30)
        points = monitor_downward_drift(values, baseline_mean=0.85, baseline_std=0.01)
        assert all(p.cusum_down >= 0.0 for p in points)

    def test_status_both_alert_when_both_triggered(self) -> None:
        """Her ikisi de tetiklenince status 'BOTH_ALERT' olmali."""
        stable = np.full(15, 0.90)
        drop = np.full(10, 0.50)
        values = np.concatenate([stable, drop])
        points = monitor_downward_drift(
            values,
            baseline_mean=0.90,
            baseline_std=0.02,
            ewma_lambda=0.5,
            cusum_h_sigma=2.0,
        )
        both_alert_points = [p for p in points if p.status == "BOTH_ALERT"]
        assert len(both_alert_points) > 0, "Dramatik dususten sonra BOTH_ALERT beklenir"


class TestClassifySuddenDrop:
    """classify_sudden_drop() icin testler."""

    def test_stable_no_drop(self) -> None:
        assert classify_sudden_drop(10.0, 10.0) == "STABLE"

    def test_small_drop_below_review(self) -> None:
        # 1.0 drop < 2.0 review threshold
        assert classify_sudden_drop(9.0, 10.0) == "STABLE"

    def test_review_threshold(self) -> None:
        # 2.0 drop == review_drop_pp -> P2_REVIEW
        assert classify_sudden_drop(8.0, 10.0) == "P2_REVIEW"

    def test_incident_threshold(self) -> None:
        # 5.0 drop == incident_drop_pp -> P1_INCIDENT
        assert classify_sudden_drop(5.0, 10.0) == "P1_INCIDENT"

    def test_blocked_threshold(self) -> None:
        # 11.0 drop > block_drop_pp=10.0 -> P0_BLOCKED
        assert classify_sudden_drop(-1.0, 10.0) == "P0_BLOCKED"

    def test_exact_review_boundary(self) -> None:
        """Esit deger o seviyeyi tetiklemeli."""
        assert classify_sudden_drop(8.0, 10.0, review_drop_pp=2.0) == "P2_REVIEW"

    def test_exact_incident_boundary(self) -> None:
        assert classify_sudden_drop(5.0, 10.0, incident_drop_pp=5.0) == "P1_INCIDENT"

    def test_exact_block_boundary(self) -> None:
        assert classify_sudden_drop(0.0, 10.0, block_drop_pp=10.0) == "P0_BLOCKED"

    def test_improvement_is_stable(self) -> None:
        """Metrik iyilesirse STABLE donmeli."""
        assert classify_sudden_drop(12.0, 10.0) == "STABLE"

    def test_custom_thresholds(self) -> None:
        """Parametre gecersiz kilma calismali."""
        result = classify_sudden_drop(
            5.0,
            10.0,
            review_drop_pp=3.0,
            incident_drop_pp=7.0,
            block_drop_pp=15.0,
        )
        # 5.0 drop: >= 3.0 (review), < 7.0 (incident)
        assert result == "P2_REVIEW"

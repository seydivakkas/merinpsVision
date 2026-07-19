"""Population Stability Index (PSI) hesaplama modulu.

Tum hesaplamalar CPU/NumPy'de yapilir. GPU bu module hicbir sekilde
dokunmaz.
"""

from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Referans ve guncel dagılım arasındaki PSI'yi hesaplar.

    Kova sinirlari referans dagilimin kantillerinden turetilir; boylece
    her referans kovasi yaklasik esit agirlik tasir.

    PSI yorumlamasi (geleneksel oneri -- ``configs/app.yaml`` ile gecersiz
    kilınabilir):

    * < 0.10 : Stabil (LOW)
    * 0.10-0.25: Dikkat (MEDIUM)
    * > 0.25 : Kritik (HIGH)

    Args:
        reference: Baslangic/referans donemindeki skor dizisi (1-D).
        current: Guncel donemindeki skor dizisi (1-D).
        bins: Kova sayisi. Varsayilan 10.
        epsilon: Sifir bolme icin kucuk stabilite sabiti.

    Returns:
        PSI degeri (float, sifir veya uzeri).

    Raises:
        ValueError: ``reference`` veya ``current`` bos oldugunda.
        ValueError: ``bins`` < 2 oldugunda.
    """
    if reference.size == 0 or current.size == 0:
        raise ValueError("reference ve current bos olamaz.")
    if bins < 2:
        raise ValueError("bins en az 2 olmalidir.")

    edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_ratio = ref_counts / reference.size + epsilon
    cur_ratio = cur_counts / current.size + epsilon

    return float(np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio)))


def psi_severity(
    psi_value: float,
    *,
    medium_threshold: float = 0.10,
    high_threshold: float = 0.25,
) -> str:
    """PSI degerini WeaveVision politika seviyesine esler.

    Esikleri ``configs/app.yaml -> drift.psi_medium_threshold`` ve
    ``drift.psi_high_threshold`` uzerinden gecersiz kilin; test setiyle
    kalibre etmeyin (proje kurali).

    Args:
        psi_value: ``population_stability_index()`` ciktisi.
        medium_threshold: LOW/MEDIUM siniri. Varsayilan 0.10.
        high_threshold: MEDIUM/HIGH siniri. Varsayilan 0.25.

    Returns:
        ``'LOW'``, ``'MEDIUM'`` veya ``'HIGH'``.

    Raises:
        ValueError: ``medium_threshold >= high_threshold`` oldugunda.
    """
    if medium_threshold >= high_threshold:
        raise ValueError("medium_threshold < high_threshold olmalidir.")
    if psi_value < medium_threshold:
        return "LOW"
    if psi_value < high_threshold:
        return "MEDIUM"
    return "HIGH"

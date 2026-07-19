"""CUSUM ve EWMA tabanli asagi yonlu drift izleme.

Tum hesaplamalar CPU/NumPy'de yapilir; GPU bu module hicbir sekilde dokunmaz.
Referans istatistikleri (baseline_mean, baseline_std) yalnizca sabit
operational-validation setinden gelmeli -- canli veriden turetilmez.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

TrendStatusLiteral = Literal["STABLE", "EWMA_ALERT", "CUSUM_ALERT", "BOTH_ALERT"]


@dataclass(frozen=True)
class TrendPoint:
    """Tek bir gozlem penceresinin drift-trend degerlendirmesi.

    Attributes:
        index: Serideki pozisyon (1'den baslar).
        value: Ham metrik degeri.
        ewma: Exponentially Weighted Moving Average.
        ewma_lower_limit: EWMA kontrol alt siniri.
        cusum_down: Asagi yonlu kumulatif toplam istatistigi.
        ewma_alert: EWMA sinirinin altina dusulmus ise True.
        cusum_alert: CUSUM karar seviyesini asmis ise True.
        status: Birlestirilmis durum etiketi.
    """

    index: int
    value: float
    ewma: float
    ewma_lower_limit: float
    cusum_down: float
    ewma_alert: bool
    cusum_alert: bool
    status: TrendStatusLiteral


def monitor_downward_drift(
    values: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    *,
    ewma_lambda: float = 0.25,
    ewma_limit_sigma: float = 3.0,
    cusum_k_sigma: float = 0.25,
    cusum_h_sigma: float = 4.0,
) -> list[TrendPoint]:
    """AP50/recall gibi metrik serilerde asagi yonlu drift izler.

    CUSUM ve EWMA hesaplari ayni taramada birlikte uretilir.
    Her iki istatistik de referans (operational-validation) istatistiklerine
    gore standardize edilmistir.

    Args:
        values: Izlenen metrik degerlerinin zaman sirasi (1-D, bos olmamali).
        baseline_mean: Operational-validation setindeki metrik ortalamasi.
        baseline_std: Operational-validation setindeki metrik std sapmasi.
        ewma_lambda: EWMA duzlestirme sabiti, (0, 1].
            Kucuk degerler gecmise daha fazla agirlik verir. Varsayilan 0.25.
        ewma_limit_sigma: Alt kontrol siniri icin std carpimi.
        cusum_k_sigma: CUSUM referans degeri icin sigma carpimi (allowance).
        cusum_h_sigma: CUSUM karar seviyesi icin sigma carpimi.

    Returns:
        Her girdi degeri icin bir ``TrendPoint`` iceren liste.

    Raises:
        ValueError: ``values`` bos ya da cok boyutlu oldugunda.
        ValueError: ``baseline_std`` <= 0 oldugunda.
        ValueError: ``ewma_lambda`` (0, 1] disinda oldugunda.

    Note:
        Bu fonksiyondaki parametre varsayilanlari baslangic politikasidir.
        Gercek esikleri ``configs/app.yaml -> drift`` bolumunden okuyun;
        test setiyle kalibre etmeyin (proje kurali).
    """
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values bos olmayan tek boyutlu bir dizi olmalidir.")
    if baseline_std <= 0.0:
        raise ValueError("baseline_std sifirdan buyuk olmalidir.")
    if not 0.0 < ewma_lambda <= 1.0:
        raise ValueError("ewma_lambda (0, 1] araliginda olmalidir.")

    ewma = baseline_mean
    cusum_down = 0.0
    points: list[TrendPoint] = []

    for index, value in enumerate(values, start=1):
        # EWMA guncelle
        ewma = ewma_lambda * float(value) + (1.0 - ewma_lambda) * ewma

        # EWMA kontrol siniri (Montgomery 2013, b. 9)
        ewma_sigma = baseline_std * np.sqrt(
            (ewma_lambda / (2.0 - ewma_lambda)) * (1.0 - (1.0 - ewma_lambda) ** (2 * index))
        )
        lower_limit = baseline_mean - ewma_limit_sigma * float(ewma_sigma)
        ewma_alert = bool(ewma < lower_limit)

        # Asagi yonlu CUSUM
        standardized_drop = (baseline_mean - float(value)) / baseline_std
        cusum_down = max(0.0, cusum_down + standardized_drop - cusum_k_sigma)
        cusum_alert = bool(cusum_down > cusum_h_sigma)

        if ewma_alert and cusum_alert:
            status: TrendStatusLiteral = "BOTH_ALERT"
        elif ewma_alert:
            status = "EWMA_ALERT"
        elif cusum_alert:
            status = "CUSUM_ALERT"
        else:
            status = "STABLE"

        points.append(
            TrendPoint(
                index=index,
                value=float(value),
                ewma=float(ewma),
                ewma_lower_limit=float(lower_limit),
                cusum_down=float(cusum_down),
                ewma_alert=ewma_alert,
                cusum_alert=cusum_alert,
                status=status,
            )
        )

    return points


def classify_sudden_drop(
    current_value: float,
    prior_value: float,
    *,
    review_drop_pp: float = 2.0,
    incident_drop_pp: float = 5.0,
    block_drop_pp: float = 10.0,
) -> str:
    """Iki ardisik pencere arasindaki ani dususu IncidentPriority'ye siniflar.

    Degerlerin ayni olcek biriminde olmasi gerekir (pp gibi mutlak birime
    gerek yok). Esikleri ``configs/app.yaml -> drift`` bolumunden okuyun.

    Args:
        current_value: Guncel pencere metrik degeri.
        prior_value: Onceki pencere metrik degeri.
        review_drop_pp: P2_REVIEW tetikleme esigi.
        incident_drop_pp: P1_INCIDENT tetikleme esigi.
        block_drop_pp: P0_BLOCKED tetikleme esigi.

    Returns:
        ``'P0_BLOCKED'``, ``'P1_INCIDENT'``, ``'P2_REVIEW'`` veya ``'STABLE'``.
    """
    drop = prior_value - current_value
    if drop >= block_drop_pp:
        return "P0_BLOCKED"
    if drop >= incident_drop_pp:
        return "P1_INCIDENT"
    if drop >= review_drop_pp:
        return "P2_REVIEW"
    return "STABLE"

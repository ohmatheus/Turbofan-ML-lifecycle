from dataclasses import dataclass


@dataclass(frozen=True)
class DriftThresholds:
    psi_warn: float = 0.1
    psi_alert: float = 0.25
    ks_warn: float = 0.1
    ks_alert: float = 0.2
    rmse_warn_factor: float = 1.1  # 10% worse than baseline
    rmse_alert_factor: float = 1.3


DEFAULT_THRESHOLDS = DriftThresholds()

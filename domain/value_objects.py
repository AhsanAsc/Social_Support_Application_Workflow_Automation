from dataclasses import dataclass


@dataclass(frozen=True)
class Money:
    amount: float
    currency: str = "AED"


@dataclass(frozen=True)
class RiskScore:
    value: float  # 0–1
    explanation: str | None = None

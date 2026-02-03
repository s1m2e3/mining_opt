from dataclasses import dataclass


@dataclass
class Pushback:
    x: float
    y: float
    z: float
    level: float
    income: float
    cost: float
    income: float
    mined: bool


@dataclass
class RelativePushback:
    x: float
    y: float
    z: float
    delta_k : int

"""Pydantic schemas for request validation and response formatting."""

import math
from typing import Dict, List, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator


class WineFeatures(BaseModel):
    """Input payload representing 13 chemical features of a wine sample."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float = Field(..., alias="od280/od315_of_diluted_wines")
    proline: float

    @field_validator("*", mode="after")
    @classmethod
    def check_finite(cls, v: float) -> float:
        """Ensure all numerical values are finite."""
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Feature value must be a finite number")
        return v

    def to_feature_list(self) -> List[float]:
        """Convert features to canonical list matching training order."""
        return [
            float(self.alcohol),
            float(self.malic_acid),
            float(self.ash),
            float(self.alcalinity_of_ash),
            float(self.magnesium),
            float(self.total_phenols),
            float(self.flavanoids),
            float(self.nonflavanoid_phenols),
            float(self.proanthocyanins),
            float(self.color_intensity),
            float(self.hue),
            float(self.od280_od315_of_diluted_wines),
            float(self.proline),
        ]

    def to_cache_key(self) -> Tuple[float, ...]:
        """Generate canonical hashable cache key tuple."""
        return tuple(self.to_feature_list())


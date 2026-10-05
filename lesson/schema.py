from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MixPart(BaseModel):
    pigment: str
    parts: int = Field(ge=1)


class Step(BaseModel):
    index: int = Field(ge=1)
    name: str
    mask_path: str
    target_rgb: tuple[int, int, int]
    mix: list[MixPart] = Field(min_length=1)
    mix_description: str = ""      # the sayable form of `mix` (layers.palette.describe_mix)
    brush: str
    technique: str
    stroke_dir_deg: int = Field(ge=0, lt=360)
    success: str


Category = Literal["value", "coverage", "none"]


class Verdict(BaseModel):
    verdict: Literal["READY", "ADJUST"]
    category: Category
    adjustment: str

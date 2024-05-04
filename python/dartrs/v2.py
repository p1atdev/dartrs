from enum import Enum

from . import dartrs


class LengthTag(Enum):
    VeryLong = dartrs.LengthTag.VeryLong
    Long = dartrs.LengthTag.Long
    Medium = dartrs.LengthTag.Medium
    Short = dartrs.LengthTag.Short
    VeryShort = dartrs.LengthTag.VeryShort


class AspectRatioTag(Enum):
    UltraWide = dartrs.AspectRatioTag.UltraWide
    Wide = dartrs.AspectRatioTag.Wide
    Square = dartrs.AspectRatioTag.Square
    Tall = dartrs.AspectRatioTag.Tall
    UltraTall = dartrs.AspectRatioTag.UltraTall


class RatingTag(Enum):
    Sfw = dartrs.RatingTag.Sfw
    General = dartrs.RatingTag.General
    Sensitive = dartrs.RatingTag.Sensitive
    Nsfw = dartrs.RatingTag.Nsfw
    Questionable = dartrs.RatingTag.Questionable
    Explicit = dartrs.RatingTag.Explicit


class IdentityTag(Enum):
    Free = dartrs.IdentityTag.Free
    Lax = dartrs.IdentityTag.Lax
    Strict = dartrs.IdentityTag.Strict


def compose_prompt(
    prompt: str = "",
    copyright: str = "",
    character: str = "",
    rating: RatingTag = RatingTag.Sfw,
    aspect_ratio: AspectRatioTag = AspectRatioTag.Tall,
    length: LengthTag = LengthTag.Long,
    identity_level: IdentityTag = IdentityTag.Lax,
):
    return dartrs.compose_prompt_v2(
        copyright=copyright,
        character=character,
        rating=rating.value,
        aspect_ratio=aspect_ratio.value,
        length=length.value,
        identity_level=identity_level.value,
        prompt=prompt,
    )


class V2Model:
    model: dartrs.DartV2Mistral | dartrs.DartV2Mixtral

    def __init__(
        self,
        model: dartrs.DartV2Mistral | dartrs.DartV2Mixtral,
    ) -> None:
        self.model = model

    def generate(self, config: dartrs.GenerationConfig) -> str:
        return self.model.generate(config)


class MixtralModel(V2Model):
    @classmethod
    def from_pretrained(
        cls,
        hub_name: str,
        revision: str | None = None,
        dtype: dartrs.DartDType = dartrs.DartDType.FP32,
        device: dartrs.DartDevice = dartrs.DartDevice.Cpu(),
    ) -> V2Model:
        return cls(dartrs.DartV2Mixtral(hub_name, revision, dtype, device))


class MistralModel(V2Model):
    @classmethod
    def from_pretrained(
        cls,
        hub_name: str,
        revision: str | None = None,
        dtype: dartrs.DartDType = dartrs.DartDType.FP32,
        device: dartrs.DartDevice = dartrs.DartDevice.Cpu(),
    ) -> V2Model:
        return cls(dartrs.DartV2Mistral(hub_name, revision, dtype, device))

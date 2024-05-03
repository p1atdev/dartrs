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


def GenerationConfig(
    prompt: str,
    tokenizer: dartrs.DartTokenizer,
    device: dartrs.DartDevice = dartrs.DartDevice.Cpu(),
    eos_token: str | None = None,
    max_new_tokens: int | None = 256,
    temperature: float | None = 1.0,
    top_k: int | None = 100,
    top_p: float | None = 0.9,
    seed: int | None = None,
) -> dartrs.DartGenerationConfig:
    return dartrs.DartGenerationConfig(
        device=device,
        tokenizer=tokenizer,
        prompt=prompt,
        eos_token=eos_token,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )


class MixtralModel:
    @classmethod
    def from_pretrained(
        cls,
        hub_name: str,
        revision: str | None = None,
        dtype: dartrs.DartDType = dartrs.DartDType.FP32,
        device: dartrs.DartDevice = dartrs.DartDevice.Cpu(),
    ) -> dartrs.DartV2Mixtral:
        return dartrs.DartV2Mixtral(hub_name, revision, dtype, device)

from typing import Literal

from . import dartrs


LengthTag = (
    Literal["<|length:very_short|>"]
    | Literal["<|length:short|>"]
    | Literal["<|length:medium|>"]
    | Literal["<|length:long|>"]
    | Literal["<|length:very_long|>"]
)

AspectRatioTag = (
    Literal["<|aspect_ratio:ultra_wide|>"]
    | Literal["<|aspect_ratio:wide|>"]
    | Literal["<|aspect_ratio:square|>"]
    | Literal["<|aspect_ratio:tall|>"]
    | Literal["<|aspect_ratio:ultra_tall|>"]
)

RatingTag = (
    Literal["<|rating:sfw|>"]
    | Literal["<|rating:general|>"]
    | Literal["<|rating:sensitive|>"]
    | Literal["<|rating:nsfw|>"]
    | Literal["<|rating:questionable|>"]
    | Literal["<|rating:explicit|>"]
)

IdentityTag = (
    Literal["<|identity:none|>"]
    | Literal["<|identity:lax|>"]
    | Literal["<|identity:strict|>"]
)


def compose_prompt(
    prompt: str = "",
    copyright: str = "",
    character: str = "",
    rating: RatingTag = "<|rating:general|>",
    aspect_ratio: AspectRatioTag = "<|aspect_ratio:tall|>",
    length: LengthTag = "<|length:medium|>",
    identity: IdentityTag = "<|identity:none|>",
):
    return dartrs.compose_prompt_v2(
        copyright=copyright,
        character=character,
        rating=dartrs.RatingTag(rating),
        aspect_ratio=dartrs.AspectRatioTag(aspect_ratio),
        length=dartrs.LengthTag(length),
        identity_level=dartrs.IdentityTag(identity),
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
        auth_token: str | None = None,
    ) -> V2Model:
        return cls(dartrs.DartV2Mixtral(hub_name, revision, dtype, device, auth_token))


class MistralModel(V2Model):
    @classmethod
    def from_pretrained(
        cls,
        hub_name: str,
        revision: str | None = None,
        dtype: dartrs.DartDType = dartrs.DartDType.FP32,
        device: dartrs.DartDevice = dartrs.DartDevice.Cpu(),
        auth_token: str | None = None,
    ) -> V2Model:
        return cls(dartrs.DartV2Mistral(hub_name, revision, dtype, device, auth_token))

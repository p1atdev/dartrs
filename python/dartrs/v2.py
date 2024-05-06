from typing import Literal

from . import dartrs


LengthTag = (
    Literal["very_short"]
    | Literal["short"]
    | Literal["medium"]
    | Literal["long"]
    | Literal["very_long"]
)

AspectRatioTag = (
    Literal["ultra_wide"]
    | Literal["wide"]
    | Literal["square"]
    | Literal["tall"]
    | Literal["ultra_tall"]
)

RatingTag = (
    Literal["sfw"]
    | Literal["general"]
    | Literal["sensitive"]
    | Literal["nsfw"]
    | Literal["questionable"]
    | Literal["explicit"]
)

IdentityTag = Literal["none"] | Literal["lax"] | Literal["strict"]


def compose_prompt(
    prompt: str = "",
    copyright: str = "",
    character: str = "",
    rating: RatingTag = "general",
    aspect_ratio: AspectRatioTag = "tall",
    length: LengthTag = "medium",
    identity: IdentityTag = "none",
    do_completion: bool = True,
):
    return dartrs.compose_prompt_v2(
        copyright=copyright,
        character=character,
        rating=dartrs.RatingTag(f"<|rating:{rating}|>"),
        aspect_ratio=dartrs.AspectRatioTag(f"<|aspect_ratio:{aspect_ratio}|>"),
        length=dartrs.LengthTag(f"<|length:{length}|>"),
        identity_level=dartrs.IdentityTag(f"<|identity:{identity}|>"),
        prompt=prompt,
        do_completion=do_completion,
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

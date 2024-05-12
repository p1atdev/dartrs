from typing import Literal, Generator

from . import dartrs
from . import utils

LengthTag = Literal["very_short", "short", "medium", "long", "very_long"]

AspectRatioTag = Literal["ultra_wide", "wide", "square", "tall", "ultra_tall"]

RatingTag = Literal["sfw", "general", "sensitive", "nsfw", "questionable", "explicit"]

IdentityTag = Literal["none", "lax", "strict"]


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
        """Generates tags."""
        return self.model.generate(config)

    def _get_next_token(
        self, config: dartrs.GenerationConfig, cache: dartrs.GenerationCache
    ) -> tuple[int, dartrs.GenerationCache]:
        """Generates the next token."""
        return self.model.get_next_token(config, cache)

    def generate_stream(self, config: dartrs.GenerationConfig) -> Generator[
        str,  # tag
        None,
        str,  # final decoded text
    ]:
        """Generates tags and returns the final decoded text."""

        tokens = config.tokenizer().encode(config.prompt())
        cache = dartrs.GenerationCache(tokens)

        for _ in range(0, config.max_new_tokens()):
            token, cache = self._get_next_token(config, cache)
            tag = config.tokenizer().decode([token], skip_special_tokens=True)
            yield tag

            if cache.finished():
                break

        self.model._clear_kv_cache()  # clear kv cache

        decoded = config.tokenizer().decode(
            cache.output_tokens(), skip_special_tokens=True
        )
        return decoded


class MixtralModel(V2Model):
    @classmethod
    def from_pretrained(
        cls,
        hub_name: str,
        revision: str | None = None,
        dtype: utils.DType = "fp32",
        device: utils.Device = "cpu",
        auth_token: str | None = None,
    ) -> V2Model:
        return cls(
            dartrs.DartV2Mixtral(
                hub_name,
                revision,
                dartrs.DartDType(dtype),
                dartrs.DartDevice(device),
                auth_token,
            )
        )


class MistralModel(V2Model):
    @classmethod
    def from_pretrained(
        cls,
        hub_name: str,
        revision: str | None = None,
        dtype: utils.DType = "fp32",
        device: utils.Device = "cpu",
        auth_token: str | None = None,
    ) -> V2Model:
        return cls(
            dartrs.DartV2Mistral(
                hub_name,
                revision,
                dartrs.DartDType(dtype),
                dartrs.DartDevice(device),
                auth_token,
            )
        )

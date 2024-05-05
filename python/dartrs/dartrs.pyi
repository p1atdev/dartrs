from abc import ABC

class DartDType:
    BF16: ...
    FP16: ...
    FP32: ...

class DartDevice:
    @classmethod
    def Cpu(cls) -> ...: ...
    @classmethod
    def Cuda(cls, id: int) -> ...: ...

class GenerationConfig:
    def __init__(
        self,
        device: DartDevice,
        tokenizer: DartTokenizer,
        prompt: str,
        eos_token: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        ban_token_ids: list[int] | None = None,
        seed: int | None = None,
    ) -> None: ...

class DartV2Mistral:
    def __init__(
        self,
        hub_name: str,
        revision: str | None = None,
        dtype: DartDType = DartDType.FP32,
        device: DartDevice = DartDevice.Cpu(),
        auth_token: str | None = None,
    ) -> None:
        raise NotImplementedError

    def generate(self, config: GenerationConfig) -> str:
        raise NotImplementedError

class DartV2Mixtral:
    def __init__(
        self,
        hub_name: str,
        revision: str | None = None,
        dtype: DartDType = DartDType.FP32,
        device: DartDevice = DartDevice.Cpu(),
        auth_token: str | None = None,
    ) -> None:
        raise NotImplementedError

    def generate(self, config: GenerationConfig) -> str:
        raise NotImplementedError

class DartTokenizer:
    @staticmethod
    def from_pretrained(
        identifier,
        revision: str | None = None,
        auth_token: str | None = None,
    ) -> DartTokenizer: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(
        self, token_ids: list[int], skip_special_tokens: bool | None = None
    ) -> str:
        """Decodes tokens and returns a the concatenated text."""
        ...

    def decode_tags(
        self, token_ids: list[int], skip_special_tokens: bool | None = None
    ) -> list[str]:
        """Decodes tokens and returns a list of tags."""
        ...

class SpecialTag(ABC):
    def __init__(self, tag: str) -> None: ...
    def to_tag(self) -> str: ...

class LengthTag(SpecialTag):
    VeryShort: LengthTag
    Short: LengthTag
    Medium: LengthTag
    Long: LengthTag
    VeryLong: LengthTag

class AspectRatioTag(SpecialTag):
    UltraWide: AspectRatioTag
    Wide: AspectRatioTag
    Square: AspectRatioTag
    Tall: AspectRatioTag
    UltraTall: AspectRatioTag

class RatingTag(SpecialTag):
    Sfw: RatingTag
    General: RatingTag
    Sensitive: RatingTag
    Nsfw: RatingTag
    Questionable: RatingTag
    Explicit: RatingTag

class IdentityTag(SpecialTag):
    Free: IdentityTag
    Lax: IdentityTag
    Strict: IdentityTag

def compose_prompt_v2(
    copyright: str,
    character: str,
    rating: RatingTag,
    aspect_ratio: AspectRatioTag,
    length: LengthTag,
    identity_level: IdentityTag,
    prompt: str,
) -> str: ...

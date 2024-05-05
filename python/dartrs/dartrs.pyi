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
        seed: int | None = None,
    ) -> None: ...

class DartV2Mistral:
    def __init__(
        self,
        hub_name: str,
        revision: str | None = None,
        dtype: DartDType = DartDType.FP32,
        device: DartDevice = DartDevice.Cpu(),
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
    ) -> None:
        raise NotImplementedError

    def generate(self, config: GenerationConfig) -> str:
        raise NotImplementedError

class DartTokenizer:
    @staticmethod
    def from_pretrained(
        identifier,
        revision=...,
    ) -> DartTokenizer: ...

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

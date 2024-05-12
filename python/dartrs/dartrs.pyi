from abc import ABC

class DartDType:
    BF16: ...
    FP16: ...
    FP32: ...
    def __init__(self, dtype: str) -> None: ...

class DartDevice:
    def __init__(self, device: str) -> None: ...
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
    def tokenizer(self) -> DartTokenizer: ...
    def prompt(self) -> str: ...
    def max_new_tokens(self) -> int: ...

class GenerationCache:
    def __init__(self, input_tokens: list[int]) -> None: ...
    def input_tokens(self) -> list[int]: ...
    def output_tokens(self) -> list[int]: ...
    def finished(self) -> bool: ...

class DartV2Mistral:
    def __init__(
        self,
        hub_name: str,
        revision: str | None = None,
        dtype: DartDType = DartDType.FP32,
        device: DartDevice = DartDevice.Cpu(),
        auth_token: str | None = None,
    ) -> None: ...
    def generate(self, config: GenerationConfig) -> str:
        raise NotImplementedError

    def get_next_token(
        self,
        config: GenerationConfig,
        cache: GenerationCache,
    ) -> tuple[int, GenerationCache]: ...
    def _clear_kv_cache(self) -> None: ...

class DartV2Mixtral:
    def __init__(
        self,
        hub_name: str,
        revision: str | None = None,
        dtype: DartDType = DartDType.FP32,
        device: DartDevice = DartDevice.Cpu(),
        auth_token: str | None = None,
    ) -> None: ...
    def generate(self, config: GenerationConfig) -> str:
        raise NotImplementedError

    def get_next_token(
        self,
        config: GenerationConfig,
        cache: GenerationCache,
    ) -> tuple[int, GenerationCache]: ...
    def _clear_kv_cache(self) -> None: ...

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

    def tokenize(self, text: str) -> list[str]:
        """Tokenizes text and returns a list of tokens."""
        ...

    def get_vocab(self, with_added_tokens: bool | None = True) -> dict[str, int]:
        """Returns the vocabulary as a dictionary. The keys are the tokens and the values are the token IDs."""
        ...

    def get_added_tokens(self) -> list[str]:
        """Returns the added tokens."""
        ...

class Tag(ABC):
    def __init__(self, tag: str) -> None: ...
    def to_tag(self) -> str: ...

class LengthTag(Tag):
    VeryShort: LengthTag
    Short: LengthTag
    Medium: LengthTag
    Long: LengthTag
    VeryLong: LengthTag

class AspectRatioTag(Tag):
    UltraWide: AspectRatioTag
    Wide: AspectRatioTag
    Square: AspectRatioTag
    Tall: AspectRatioTag
    UltraTall: AspectRatioTag

class RatingTag(Tag):
    Sfw: RatingTag
    General: RatingTag
    Sensitive: RatingTag
    Nsfw: RatingTag
    Questionable: RatingTag
    Explicit: RatingTag

class IdentityTag(Tag):
    Free: IdentityTag
    Lax: IdentityTag
    Strict: IdentityTag

class SpecialTag(Tag):
    Bos: SpecialTag
    Eos: SpecialTag
    CopyrightStart: SpecialTag
    CopyrightEnd: SpecialTag
    CharacterStart: SpecialTag
    CharacterEnd: SpecialTag
    GeneralStart: SpecialTag
    GeneralEnd: SpecialTag
    InputEnd: SpecialTag

def compose_prompt_v2(
    copyright: str,
    character: str,
    rating: RatingTag,
    aspect_ratio: AspectRatioTag,
    length: LengthTag,
    identity_level: IdentityTag,
    prompt: str,
    do_completion: bool,
) -> str: ...

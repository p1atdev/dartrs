from typing import Optional, Callable
from tokenizers import Tokenizer

class DartDType:
    BF16: ...
    FP16: ...
    FP32: ...

class DartDevice:
    @classmethod
    def Cpu(cls) -> ...: ...
    @classmethod
    def Cuda(cls, id: int) -> ...: ...

class DartGenerationConfig:
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

class DartModel:
    def __init__(
        self,
        hub_name: str,
        revision: str | None = None,
        dtype: DartDType = DartDType.FP32,
        device: DartDevice = DartDevice.Cpu(),
    ) -> None:
        raise NotImplementedError

    def generate(self, config: DartGenerationConfig) -> str:
        raise NotImplementedError

class DartV2Mistral(DartModel): ...
class DartV2Mixtral(DartModel): ...

class DartTokenizer:
    @staticmethod
    def from_pretrained(
        identifier,
        revision=...,
    ) -> DartTokenizer: ...

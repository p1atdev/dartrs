from dartrs.dartrs import (
    DartV2Mistral,
    DartV2Mixtral,
    DartDevice,
    DartDType,
    DartTokenizer,
    DartGenerationConfig,
)


def test_device():
    device = DartDevice.Cpu()
    assert device is not None

    device = DartDevice.Cuda(0)
    assert device is not None


def test_tokenizer():
    tokenizer = DartTokenizer.from_pretrained("p1atdev/dart-v2-mixtral-160m-sft-2")

    assert tokenizer is not None


def test_generation_config():
    config = DartGenerationConfig(
        device=DartDevice.Cpu(),
        tokenizer=DartTokenizer.from_pretrained("p1atdev/dart-v2-mixtral-160m-sft-2"),
        prompt="<|bos|><copyright></copyright><character></character><|rating:sfw|><|aspect_ratio:tall|><|length:long|><general>1girl<|identity:lax|><|input_end|>",
    )

    assert config is not None


def test_mistral_model():
    model_name = "p1atdev/dart-v2-mistral-100m-sft"

    model = DartV2Mistral(model_name)
    assert model is not None


def test_mixtral_model():
    model_name = "p1atdev/dart-v2-mixtral-160m-sft-2"

    model = DartV2Mixtral(model_name)
    assert model is not None

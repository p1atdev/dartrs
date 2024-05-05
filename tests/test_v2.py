from dartrs.v2 import MistralModel, MixtralModel
from dotenv import load_dotenv
import os

load_dotenv()


def test_v2_mistral_model():
    model = MistralModel.from_pretrained("p1atdev/dart-v2-mistral-100m-sft")

    assert model is not None


def test_v2_mixtral_model():
    model = MixtralModel.from_pretrained("p1atdev/dart-v2-mixtral-160m-sft-2")

    assert model is not None


def test_v2_mistral_model_with_auth_token():
    TEST_HF_TOKEN = os.getenv("TEST_HF_TOKEN")

    model = MistralModel.from_pretrained(
        "p1atdev/dart-v2-mistral-100m-sft", auth_token=TEST_HF_TOKEN
    )

    assert model is not None

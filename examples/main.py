from dartrs.dartrs import DartDevice, DartTokenizer, GenerationConfig
from dartrs.utils import get_generation_config
from dartrs.v2 import (
    compose_prompt,
    MixtralModel,
    V2Model,
)
import time

MODEL_NAME = "p1atdev/dart-v2-mixtral-160m-sft-8"


def prepare_models():
    model = MixtralModel.from_pretrained(MODEL_NAME)
    tokenizer = DartTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer


def generate(model: V2Model, config: GenerationConfig):
    start = time.time()
    output = model.generate(config)
    end = time.time()

    print(output)
    print(f"Time taken: {end - start:.2f}s")


def main():
    model, tokenizer = prepare_models()

    # generate 5 times
    generate(
        model,
        get_generation_config(
            prompt=compose_prompt(
                copyright="",
                character="",
                rating="<|rating:general|>",
                aspect_ratio="<|aspect_ratio:tall|>",
                length="<|length:medium|>",
                identity="<|identity:none|>",
                prompt="1girl, cat ears",
            ),
            tokenizer=tokenizer,
            seed=42,
        ),
    )
    generate(
        model,
        get_generation_config(
            prompt=compose_prompt(
                prompt="2girls",
            ),
            tokenizer=tokenizer,
            seed=999999,
        ),
    )

    generate(
        model,
        get_generation_config(
            prompt=compose_prompt(
                prompt="1girl, solo",
            ),
            tokenizer=tokenizer,
        ),
    )


if __name__ == "__main__":
    main()

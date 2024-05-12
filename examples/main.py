from dartrs.dartrs import DartTokenizer, GenerationConfig, DartDType
from dartrs.utils import get_generation_config
from dartrs.v2 import (
    compose_prompt,
    MixtralModel,
    V2Model,
)
import time
import os

MODEL_NAME = "p1atdev/dart-v2-moe-sft"


def prepare_models():
    model = MixtralModel.from_pretrained(MODEL_NAME, dtype="fp16")
    tokenizer = DartTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer


def generate(model: V2Model, config: GenerationConfig):
    start = time.time()
    output = model.generate(config)
    end = time.time()

    print(output)
    print(f"Time taken: {end - start:.2f}s")


def generate_stream(model: V2Model, config: GenerationConfig):
    start = time.time()
    for tag in model.generate_stream(config):
        if tag.strip() == "":
            continue
        os.write(1, tag.encode("utf-8") + b", ")
    end = time.time()

    print()
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
                rating="general",
                aspect_ratio="tall",
                length="medium",
                identity="none",
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

    generate_stream(
        model,
        get_generation_config(
            prompt=compose_prompt(
                prompt="1girl, solo",
                length="long",
            ),
            tokenizer=tokenizer,
        ),
    )


if __name__ == "__main__":
    main()

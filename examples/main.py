from dartrs.dartrs import DartDevice, DartTokenizer, DartModel, DartGenerationConfig
from dartrs.v2 import (
    compose_prompt,
    GenerationConfig,
    LengthTag,
    RatingTag,
    AspectRatioTag,
    IdentityTag,
    MixtralModel,
)
import time

MODEL_NAME = "p1atdev/dart-v2-mixtral-160m-sft-8"


def prepare_models():
    model = MixtralModel.from_pretrained(MODEL_NAME)
    tokenizer = DartTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer


def generate(model: DartModel, config: DartGenerationConfig):
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
        GenerationConfig(
            prompt=compose_prompt(
                copyright="",
                character="",
                rating=RatingTag.Sfw,
                aspect_ratio=AspectRatioTag.Tall,
                length=LengthTag.Long,
                identity_level=IdentityTag.Free,
                prompt="1girl, cat ears",
            ),
            tokenizer=tokenizer,
            seed=42,
        ),
    )
    generate(
        model,
        GenerationConfig(
            prompt=compose_prompt(
                prompt="2girls",
            ),
            tokenizer=tokenizer,
            seed=999999,
        ),
    )

    generate(
        model,
        GenerationConfig(
            prompt=compose_prompt(
                prompt="1girl, solo",
            ),
            tokenizer=tokenizer,
        ),
    )


if __name__ == "__main__":
    main()

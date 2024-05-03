from dartrs.dartrs import (
    DartV2Mixtral,
    DartDevice,
    DartGenerationConfig,
    DartTokenizer,
    RatingTag,
    AspectRatioTag,
    LengthTag,
    IdentityTag,
    compose_prompt_v2,
)
import time

MODEL_NAME = "p1atdev/dart-v2-mixtral-160m-sft-2"

model = DartV2Mixtral(MODEL_NAME)
print(model)

tokenizer = DartTokenizer.from_pretrained(MODEL_NAME)

generation_config = DartGenerationConfig(
    device=DartDevice.Cpu(),
    tokenizer=tokenizer,
    prompt=compose_prompt_v2(
        copyright="",
        character="",
        rating=RatingTag.Sfw,
        aspect_ratio=AspectRatioTag.Tall,
        length=LengthTag.Long,
        identity_level=IdentityTag.Lax,
        prompt="1girl, cat ears",
    ),
)

start = time.time()
output = model.generate(generation_config)
end = time.time()

print(output)
print(f"Time taken: {end - start:.2f}s")

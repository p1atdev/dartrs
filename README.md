# dartrs

> [!WARNING]
> wip

## Python

```bash
pip install -U dartrs
```

```py
from dartrs.dartrs import DartTokenizer
from dartrs.utils import get_generation_config
from dartrs.v2 import (
    compose_prompt,
    MixtralModel,
    V2Model,
)
import time
import os

MODEL_NAME = "" # TODO

model = MixtralModel.from_pretrained(MODEL_NAME)
tokenizer = DartTokenizer.from_pretrained(MODEL_NAME)

config = get_generation_config(
    prompt=compose_prompt(
        copyright="vocaloid",
        character="hatsune miku",
        rating="general", # sfw, general, sensitive, nsfw, questionable, explicit
        aspect_ratio="tall", # ultra_wide, wide, square, tall, ultra_tall
        length="medium", # very_short, short, medium, long, very_long
        identity="none", # none, lax, strict
        prompt="1girl, cat ears",
    ),
    tokenizer=tokenizer,
    seed=42,
)

start = time.time()
output = model.generate(config)
end = time.time()

print(output)
print(f"Time taken: {end - start:.2f}s")
# Output:
# cowboy shot, detached sleeves, empty eyes, green eyes, green hair, green necktie, hair in own mouth, hair ornament, letterboxed, light frown, long hair, long sleeves, looking to the side, necktie, parted lips, shirt, sleeveless, sleeveless shirt, twintails, wing collar
# Time taken: 0.26s
```
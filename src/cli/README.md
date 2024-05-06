# dartrs cli example

to run:

```bash
cargo run --release -- -p "1girl" --model-type mixtral --model-name "p1atdev/dart-v2-mixtral-160m-sft-8"
```

> [!NOTE]
> If `--release` flag is not set, it will take a very long time to generate tags.

the result:

```
avx: false, neon: false, simd128: false, f16c: false
loaded the model in 226.4033ms
<|bos|><copyright></copyright><character></character><|rating:sfw|><|aspect_ratio:tall|><|length:long|><general>1girl<|identity:lax|><|input_end|>ass, bare arms, bare shoulders, blonde hair, blue eyes, blue one-piece swimsuit, blue sky, blush, breasts, brown hair, cloud, competition school swimsuit, day, goggles, goggles on head, half-closed eyes, looking at viewer, medium breasts, mole, mole under eye, one-piece swimsuit, open mouth, outdoors, ponytail, pool, school swimsuit, shiny clothes, shiny skin, sky, swim cap, swimsuit, thighs, wet, wet clothes, white headwear, </general>, <|eos|>,
37 tokens generated (102.58 token/s)
````


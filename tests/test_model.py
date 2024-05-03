from dartrs.dartrs import (
    DartDevice,
    DartTokenizer,
)
from dartrs.v2 import (
    MixtralModel,
    GenerationConfig,
    compose_prompt,
    LengthTag,
    RatingTag,
    AspectRatioTag,
    IdentityTag,
)
from random import randint


def prepare_models():
    model = MixtralModel.from_pretrained("p1atdev/dart-v2-mixtral-160m-sft-2")
    tokenizer = DartTokenizer.from_pretrained("p1atdev/dart-v2-mixtral-160m-sft-2")

    return model, tokenizer


def test_generate():
    model, tokenizer = prepare_models()

    prompt = compose_prompt(
        prompt="1girl, cat ears",
    )
    config = GenerationConfig(
        prompt=prompt,
        tokenizer=tokenizer,
        seed=42,
    )

    output = model.generate(config)
    assert output is not None


def test_generate_different_seed():
    model, tokenizer = prepare_models()

    seeds = [1, 2900]
    results = []

    for seed in seeds:
        prompt = compose_prompt(
            prompt="1girl, cat ears",
        )
        config = GenerationConfig(
            prompt=prompt,
            tokenizer=tokenizer,
            seed=seed,
        )

        results.append(model.generate(config))

    assert results[0] != results[1]


def test_generate_random_seed():
    model, tokenizer = prepare_models()

    seeds = [randint(0, 10000) for _ in range(10)]
    results = []

    for seed in seeds:
        prompt = compose_prompt(
            prompt="1girl, cat ears",
        )
        config = GenerationConfig(
            prompt=prompt,
            tokenizer=tokenizer,
            seed=seed,
        )

        results.append(model.generate(config))

    for i in range(1, len(results)):
        assert results[i] != results[i - 1]


def test_generate_same_seed():
    model, tokenizer = prepare_models()

    seed = 12345

    prompt = compose_prompt(
        prompt="1girl, cat ears",
    )
    config = GenerationConfig(
        prompt=prompt,
        tokenizer=tokenizer,
        seed=seed,
    )

    output1 = model.generate(config)
    output2 = model.generate(config)

    assert output1 == output2


def test_generate_different_temperature():
    model, tokenizer = prepare_models()

    temperatures = [0.5, 1.0, 1.5]
    results = []

    for temperature in temperatures:
        prompt = compose_prompt(
            prompt="1girl, cat ears",
        )
        config = GenerationConfig(
            prompt=prompt,
            tokenizer=tokenizer,
            seed=42,
            temperature=temperature,
        )

        results.append(model.generate(config))

    assert results[0] != results[1]
    assert results[1] != results[2]
    assert results[0] != results[2]


def test_generate_different_top_p():
    model, tokenizer = prepare_models()

    top_ps = [0.5, 0.9, 0.99]
    results = []

    for top_p in top_ps:
        prompt = compose_prompt(
            prompt="1girl, cat ears",
        )
        config = GenerationConfig(
            prompt=prompt,
            tokenizer=tokenizer,
            seed=42,
            top_p=top_p,
        )

        results.append(model.generate(config))

    assert results[0] != results[1]
    assert results[1] != results[2]
    assert results[0] != results[2]


def test_generate_different_top_k():
    model, tokenizer = prepare_models()

    top_ks = [50, 100, 200]
    results = []

    for top_k in top_ks:
        prompt = compose_prompt(
            prompt="1girl, cat ears",
        )
        config = GenerationConfig(
            prompt=prompt,
            tokenizer=tokenizer,
            seed=42,
            top_k=top_k,
        )

        results.append(model.generate(config))

    assert results[0] != results[1]
    assert results[1] != results[2]
    assert results[0] != results[2]


def test_generate_different_prompt():
    model, tokenizer = prepare_models()

    prompts = [
        compose_prompt(
            prompt="1girl, cat ears",
        ),
        compose_prompt(
            prompt="2girls",
        ),
        compose_prompt(
            prompt="no humans, scenery",
        ),
    ]

    results = []

    for prompt in prompts:
        config = GenerationConfig(
            prompt=prompt,
            tokenizer=tokenizer,
            seed=42,
        )

        results.append(model.generate(config))

    assert results[0] != results[1]
    assert results[1] != results[2]

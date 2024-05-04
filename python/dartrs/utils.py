from . import dartrs


def get_generation_config(
    prompt: str,
    tokenizer: dartrs.DartTokenizer,
    device: dartrs.DartDevice = dartrs.DartDevice.Cpu(),
    eos_token: str | None = None,
    max_new_tokens: int | None = 256,
    temperature: float | None = 1.0,
    top_p: float | None = 1.0,
    top_k: int | None = 100,
    seed: int | None = None,
) -> dartrs.GenerationConfig:
    return dartrs.GenerationConfig(
        device=device,
        tokenizer=tokenizer,
        prompt=prompt,
        eos_token=eos_token,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )

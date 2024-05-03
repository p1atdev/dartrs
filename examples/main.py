from dartrs.dartrs import DartV2Mixtral, DartDevice, DartGenerationConfig, DartTokenizer
import time

MODEL_NAME = "p1atdev/dart-v2-mixtral-160m-sft-4"

model = DartV2Mixtral(MODEL_NAME)
print(model)

tokenizer = DartTokenizer.from_pretrained(MODEL_NAME)

generation_config = DartGenerationConfig(
    device=DartDevice.Cpu(),
    tokenizer=tokenizer,
    prompt="<|bos|><copyright></copyright><character></character><|rating:sfw|><|aspect_ratio:tall|><|length:long|><general>1girl<|identity:lax|><|input_end|>",
)

start = time.time()
output = model.generate(generation_config)
end = time.time()

print(output)
print(f"Time taken: {end - start:.2f}s")

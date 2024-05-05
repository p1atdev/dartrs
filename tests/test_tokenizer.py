from dartrs.dartrs import DartTokenizer


def test_load_tokenizer():
    tokenizer = DartTokenizer.from_pretrained("p1atdev/dart-v2-mixtral-160m-sft-2")
    assert tokenizer is not None


def test_encode_text():
    tokenizer = DartTokenizer.from_pretrained("p1atdev/dart-v2-mixtral-160m-sft-2")

    text = "1girl, cat ears"
    encoded = tokenizer.encode(text)

    assert encoded is not None
    assert len(encoded) > 0
    assert encoded == [103, 255]


def test_decode_text():
    tokenizer = DartTokenizer.from_pretrained("p1atdev/dart-v2-mixtral-160m-sft-2")

    text = "1girl, cat ears</general><|eos|>"
    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode(encoded)

    assert decoded is not None
    assert decoded == "1girl, cat ears"


def test_decode_tags():
    tokenizer = DartTokenizer.from_pretrained("p1atdev/dart-v2-mixtral-160m-sft-2")

    text = "1girl, cat ears</general><|eos|>"
    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode_tags(encoded)

    assert decoded is not None
    assert decoded == ["1girl", "cat ears"]


def test_decode_tags_text_with_special_tokens():
    tokenizer = DartTokenizer.from_pretrained("p1atdev/dart-v2-mixtral-160m-sft-2")

    text = "1girl, cat ears</general><|eos|>"
    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode_tags(encoded, skip_special_tokens=False)

    assert decoded is not None
    assert decoded == ["1girl", "cat ears", "</general>", "<|eos|>"]

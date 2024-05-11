from dartrs.dartrs import DartTokenizer

TOKENIZER_NAME = "p1atdev/dart-v2-moe-sft"


def test_load_tokenizer():
    tokenizer = DartTokenizer.from_pretrained(TOKENIZER_NAME)
    assert tokenizer is not None


def test_encode_text():
    tokenizer = DartTokenizer.from_pretrained(TOKENIZER_NAME)

    text = "1girl, cat ears"
    encoded = tokenizer.encode(text)

    assert encoded is not None
    assert len(encoded) > 0
    assert encoded == [103, 255]


def test_decode_text():
    tokenizer = DartTokenizer.from_pretrained(TOKENIZER_NAME)

    text = "1girl, cat ears</general><|eos|>"
    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode(encoded)

    assert decoded is not None
    assert decoded == "1girl, cat ears"


def test_decode_tags():
    tokenizer = DartTokenizer.from_pretrained(TOKENIZER_NAME)

    text = "1girl, cat ears</general><|eos|>"
    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode_tags(encoded)

    assert decoded is not None
    assert decoded == ["1girl", "cat ears"]


def test_decode_tags_text_with_special_tokens():
    tokenizer = DartTokenizer.from_pretrained(TOKENIZER_NAME)

    text = "1girl, cat ears</general><|eos|>"
    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode_tags(encoded, skip_special_tokens=False)

    assert decoded is not None
    assert decoded == ["1girl", "cat ears", "</general>", "<|eos|>"]


def test_tokenize_text():
    tokenizer = DartTokenizer.from_pretrained(TOKENIZER_NAME)

    text = "1girl, cat ears, hogeeee"
    tokens = tokenizer.tokenize(text)

    assert tokens is not None
    assert tokens == ["1girl", "cat ears", "<|unk|>"]


def test_get_vocab():
    tokenizer = DartTokenizer.from_pretrained(TOKENIZER_NAME)

    vocab = tokenizer.get_vocab(True)
    assert vocab is not None
    assert "<|unk|>" in vocab

    vocab_without_added_tokens = tokenizer.get_vocab(False)
    assert vocab_without_added_tokens is not None

    assert "<|unk|>" not in vocab_without_added_tokens
    assert len(vocab) > len(vocab_without_added_tokens)


def test_get_added_tokens():
    tokenizer = DartTokenizer.from_pretrained(TOKENIZER_NAME)

    added_tokens = tokenizer.get_added_tokens()
    assert added_tokens is not None
    assert len(added_tokens) > 0

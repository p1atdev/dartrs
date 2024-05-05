from dartrs import dartrs, v2


def test_compose_prompt_v2():
    prompt = dartrs.compose_prompt_v2(
        copyright="vocaloid",
        character="hatsune miku",
        rating=dartrs.RatingTag.Sfw,
        aspect_ratio=dartrs.AspectRatioTag.Tall,
        length=dartrs.LengthTag.Long,
        identity_level=dartrs.IdentityTag.Lax,
        prompt="1girl, cat ears",
    )

    assert prompt is not None
    assert prompt == (
        f"<|bos|>"
        f"<copyright>vocaloid</copyright>"
        f"<character>hatsune miku</character>"
        f"<|rating:sfw|><|aspect_ratio:tall|><|length:long|>"
        f"<general>1girl, cat ears<|identity:lax|><|input_end|>"
    )


def test_compose_prompt_v2_wrapper():
    prompt = v2.compose_prompt(
        prompt="1girl, cat ears",
        copyright="vocaloid",
        character="hatsune miku",
        rating="<|rating:sfw|>",
        aspect_ratio="<|aspect_ratio:tall|>",
        length="<|length:long|>",
        identity="<|identity:lax|>",
    )

    assert prompt is not None
    assert prompt == (
        f"<|bos|>"
        f"<copyright>vocaloid</copyright>"
        f"<character>hatsune miku</character>"
        f"<|rating:sfw|><|aspect_ratio:tall|><|length:long|>"
        f"<general>1girl, cat ears<|identity:lax|><|input_end|>"
    )

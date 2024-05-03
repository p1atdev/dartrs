from dartrs.dartrs import (
    compose_prompt_v2,
    RatingTag,
    AspectRatioTag,
    LengthTag,
    IdentityTag,
)


def test_compose_prompt_v2():
    prompt = compose_prompt_v2(
        copyright="vocaloid",
        character="hatsune miku",
        rating=RatingTag.Sfw,
        aspect_ratio=AspectRatioTag.Tall,
        length=LengthTag.Long,
        identity_level=IdentityTag.Lax,
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

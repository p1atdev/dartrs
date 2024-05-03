use crate::tags::ReservedTag::{
    Bos, CharacterEnd, CharacterStart, CopyrightEnd, CopyrightStart, GeneralStart, InputEnd,
};
use crate::tags::{AspectRatioTag, IdentityTag, LengthTag, RatingTag, SpecialTag};

pub fn compose_prompt_v2(
    copyright: &str,
    character: &str,
    rating: RatingTag,
    aspect_ratio: AspectRatioTag,
    length: LengthTag,
    identity_level: IdentityTag,
    prompt: &str,
) -> String {
    let rating = rating.to_tag();
    let aspect_ratio = aspect_ratio.to_tag();
    let length = length.to_tag();
    let identity_level = identity_level.to_tag();

    format!(
        "\
{Bos}\
{CopyrightStart}{copyright}{CopyrightEnd}\
{CharacterStart}{character}{CharacterEnd}\
{rating}{aspect_ratio}{length}\
{GeneralStart}{prompt}{identity_level}{InputEnd}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compose_prompt() {
        let prompt = compose_prompt_v2(
            "vocaloid",
            "hatsune miku",
            RatingTag::Sfw,
            AspectRatioTag::Tall,
            LengthTag::Long,
            IdentityTag::None,
            "1girl, blue hair",
        );

        assert_eq!(
            prompt,
            r"\
<|bos|>\
<copyright>vocaloid</copyright>\
<character>hatsune miku</character>\
<|rating:sfw|><|aspect_ratio:tall|><|length:long|>\
<general>1girl, blue hair<|identity:none|><|input_end|>"
        )
    }
}

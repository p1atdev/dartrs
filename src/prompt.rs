use crate::tags::SpecialTag::{
    Bos, CharacterEnd, CharacterStart, CopyrightEnd, CopyrightStart, GeneralStart, InputEnd,
};
use crate::tags::{AspectRatioTag, IdentityTag, LengthTag, RatingTag, Tag};

pub fn compose_prompt_v2(
    copyright: &str,
    character: &str,
    rating: RatingTag,
    aspect_ratio: AspectRatioTag,
    length: LengthTag,
    identity_level: IdentityTag,
    prompt: &str,
    do_completion: bool,
) -> String {
    let rating = rating.to_tag();
    let aspect_ratio = aspect_ratio.to_tag();
    let length = length.to_tag();
    let identity_level = identity_level.to_tag();

    if do_completion {
        format!(
            "\
{Bos}\
{CopyrightStart}{copyright}{CopyrightEnd}\
{CharacterStart}{character}{CharacterEnd}\
{rating}{aspect_ratio}{length}\
{GeneralStart}{prompt}{identity_level}{InputEnd}"
        )
    } else {
        format!(
            "\
{Bos}\
{CopyrightStart}{copyright}{CopyrightEnd}\
{CharacterStart}{character}{CharacterEnd}\
{rating}{aspect_ratio}{length}\
{GeneralStart}{prompt}"
        )
    }
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
            true,
        );

        assert_eq!(
            prompt,
            "\
<|bos|>\
<copyright>vocaloid</copyright>\
<character>hatsune miku</character>\
<|rating:sfw|><|aspect_ratio:tall|><|length:long|>\
<general>1girl, blue hair<|identity:none|><|input_end|>"
        )
    }

    #[test]
    fn test_compose_prompt_no_completion() {
        let prompt = compose_prompt_v2(
            "vocaloid",
            "hatsune miku",
            RatingTag::Sfw,
            AspectRatioTag::Tall,
            LengthTag::Long,
            IdentityTag::None,
            "1girl",
            false,
        );

        assert_eq!(
            prompt,
            "\
<|bos|>\
<copyright>vocaloid</copyright>\
<character>hatsune miku</character>\
<|rating:sfw|><|aspect_ratio:tall|><|length:long|>\
<general>1girl"
        )
    }
}

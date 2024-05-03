use crate::bindings::tags::{DartAspectRatioTag, DartIdentityTag, DartLengthTag, DartRatingTag};
use crate::prompt::compose_prompt_v2;

use crate::tags::{AspectRatioTag, IdentityTag, LengthTag, RatingTag};

use pyo3::prelude::*;

#[pyfunction(name = "compose_prompt_v2")]
pub fn dart_compose_prompt_v2(
    copyright: &str,
    character: &str,
    rating: DartRatingTag,
    aspect_ratio: DartAspectRatioTag,
    length: DartLengthTag,
    identity_level: DartIdentityTag,
    prompt: &str,
) -> String {
    let rating = RatingTag::from(rating);
    let aspect_ratio = AspectRatioTag::from(aspect_ratio);
    let length = LengthTag::from(length);
    let identity_level = IdentityTag::from(identity_level);

    compose_prompt_v2(
        &copyright,
        &character,
        rating,
        aspect_ratio,
        length,
        identity_level,
        &prompt,
    )
}

use crate::tags::{AspectRatioTag, IdentityTag, LengthTag, RatingTag, ReservedTag, SpecialTag};

use pyo3::exceptions;
use pyo3::prelude::*;

#[pyclass(name = "LengthTag")]
#[derive(Debug, Clone)]
pub enum DartLengthTag {
    VeryShort,
    Short,
    Medium,
    Long,
    VeryLong,
}

impl From<DartLengthTag> for LengthTag {
    fn from(tag: DartLengthTag) -> Self {
        match tag {
            DartLengthTag::VeryShort => LengthTag::VeryShort,
            DartLengthTag::Short => LengthTag::Short,
            DartLengthTag::Medium => LengthTag::Medium,
            DartLengthTag::Long => LengthTag::Long,
            DartLengthTag::VeryLong => LengthTag::VeryLong,
        }
    }
}

#[pymethods]
impl DartLengthTag {
    #[new]
    fn new(tag: &str) -> PyResult<Self> {
        match tag {
            "<|length:very_short|>" => Ok(DartLengthTag::VeryShort),
            "<|length:short|>" => Ok(DartLengthTag::Short),
            "<|length:medium|>" => Ok(DartLengthTag::Medium),
            "<|length:long|>" => Ok(DartLengthTag::Long),
            "<|length:very_long|>" => Ok(DartLengthTag::VeryLong),
            _ => Err(exceptions::PyValueError::new_err("invalid length tag")),
        }
    }

    fn to_tag(&self) -> String {
        LengthTag::from(self.clone()).to_tag()
    }
}

#[pyclass(name = "AspectRatioTag")]
#[derive(Debug, Clone)]
pub enum DartAspectRatioTag {
    UltraWide,
    Wide,
    Square,
    Tall,
    UltraTall,
}

impl From<DartAspectRatioTag> for AspectRatioTag {
    fn from(tag: DartAspectRatioTag) -> Self {
        match tag {
            DartAspectRatioTag::UltraWide => AspectRatioTag::UltraWide,
            DartAspectRatioTag::Wide => AspectRatioTag::Wide,
            DartAspectRatioTag::Square => AspectRatioTag::Square,
            DartAspectRatioTag::Tall => AspectRatioTag::Tall,
            DartAspectRatioTag::UltraTall => AspectRatioTag::UltraTall,
        }
    }
}

#[pymethods]
impl DartAspectRatioTag {
    #[new]
    fn new(tag: &str) -> PyResult<Self> {
        match tag {
            "<|aspect_ratio:ultra_wide|>" => Ok(DartAspectRatioTag::UltraWide),
            "<|aspect_ratio:wide|>" => Ok(DartAspectRatioTag::Wide),
            "<|aspect_ratio:square|>" => Ok(DartAspectRatioTag::Square),
            "<|aspect_ratio:tall|>" => Ok(DartAspectRatioTag::Tall),
            "<|aspect_ratio:ultra_tall|>" => Ok(DartAspectRatioTag::UltraTall),
            _ => Err(exceptions::PyValueError::new_err(
                "invalid aspect ratio tag",
            )),
        }
    }

    fn to_tag(&self) -> String {
        AspectRatioTag::from(self.clone()).to_tag()
    }
}

#[pyclass(name = "RatingTag")]
#[derive(Debug, Clone)]
pub enum DartRatingTag {
    Sfw,
    General,
    Sensitive,
    Nsfw,
    Questionable,
    Explicit,
}

impl From<DartRatingTag> for RatingTag {
    fn from(tag: DartRatingTag) -> Self {
        match tag {
            DartRatingTag::Sfw => RatingTag::Sfw,
            DartRatingTag::General => RatingTag::General,
            DartRatingTag::Sensitive => RatingTag::Sensitive,
            DartRatingTag::Nsfw => RatingTag::Nsfw,
            DartRatingTag::Questionable => RatingTag::Questionable,
            DartRatingTag::Explicit => RatingTag::Explicit,
        }
    }
}

#[pymethods]
impl DartRatingTag {
    #[new]
    fn new(tag: &str) -> PyResult<Self> {
        match tag {
            "<|rating:sfw|>" => Ok(DartRatingTag::Sfw),
            "<|rating:general|>" => Ok(DartRatingTag::General),
            "<|rating:sensitive|>" => Ok(DartRatingTag::Sensitive),
            "<|rating:nsfw|>" => Ok(DartRatingTag::Nsfw),
            "<|rating:questionable|>" => Ok(DartRatingTag::Questionable),
            "<|rating:explicit|>" => Ok(DartRatingTag::Explicit),
            _ => Err(exceptions::PyValueError::new_err("invalid rating tag")),
        }
    }

    fn to_tag(&self) -> String {
        RatingTag::from(self.clone()).to_tag()
    }
}

#[pyclass(name = "IdentityTag")]
#[derive(Debug, Clone)]
pub enum DartIdentityTag {
    Free,
    Lax,
    Strict,
}

impl From<DartIdentityTag> for IdentityTag {
    fn from(tag: DartIdentityTag) -> Self {
        match tag {
            DartIdentityTag::Free => IdentityTag::None,
            DartIdentityTag::Lax => IdentityTag::Lax,
            DartIdentityTag::Strict => IdentityTag::Strict,
        }
    }
}

#[pymethods]
impl DartIdentityTag {
    #[new]
    fn new(tag: &str) -> PyResult<Self> {
        match tag {
            "<|identity:none|>" => Ok(DartIdentityTag::Free),
            "<|identity:lax|>" => Ok(DartIdentityTag::Lax),
            "<|identity:strict|>" => Ok(DartIdentityTag::Strict),
            _ => Err(exceptions::PyValueError::new_err("invalid identity tag")),
        }
    }

    fn to_tag(&self) -> String {
        IdentityTag::from(self.clone()).to_tag()
    }
}

#[pyclass(name = "ReservedTag")]
#[derive(Debug, Clone)]
pub enum DartReservedTag {
    Bos,
    Eos,
    CopyrightStart,
    CopyrightEnd,
    CharacterStart,
    CharacterEnd,
    GeneralStart,
    GeneralEnd,
    InputEnd,
}

impl From<DartReservedTag> for ReservedTag {
    fn from(tag: DartReservedTag) -> Self {
        match tag {
            DartReservedTag::Bos => ReservedTag::Bos,
            DartReservedTag::Eos => ReservedTag::Eos,
            DartReservedTag::CopyrightStart => ReservedTag::CopyrightStart,
            DartReservedTag::CopyrightEnd => ReservedTag::CopyrightEnd,
            DartReservedTag::CharacterStart => ReservedTag::CharacterStart,
            DartReservedTag::CharacterEnd => ReservedTag::CharacterEnd,
            DartReservedTag::GeneralStart => ReservedTag::GeneralStart,
            DartReservedTag::GeneralEnd => ReservedTag::GeneralEnd,
            DartReservedTag::InputEnd => ReservedTag::InputEnd,
        }
    }
}

#[pymethods]
impl DartReservedTag {
    #[new]
    fn new(tag: &str) -> PyResult<Self> {
        match tag {
            "<|bos|>" => Ok(DartReservedTag::Bos),
            "<|eos|>" => Ok(DartReservedTag::Eos),
            "<copyright>" => Ok(DartReservedTag::CopyrightStart),
            "</copyright>" => Ok(DartReservedTag::CopyrightEnd),
            "<character>" => Ok(DartReservedTag::CharacterStart),
            "</character>" => Ok(DartReservedTag::CharacterEnd),
            "<general>" => Ok(DartReservedTag::GeneralStart),
            "</general>" => Ok(DartReservedTag::GeneralEnd),
            "<|input_end|>" => Ok(DartReservedTag::InputEnd),
            _ => Err(exceptions::PyValueError::new_err("invalid reserved tag")),
        }
    }

    fn to_tag(&self) -> String {
        ReservedTag::from(self.clone()).to_tag()
    }
}

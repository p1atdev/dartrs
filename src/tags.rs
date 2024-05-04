use anyhow::{Error as E, Result};
use std::fmt;
use std::str::FromStr;

pub trait SpecialTag {
    fn to_tag(&self) -> String;
    fn is_special(tag: &str) -> bool;
}

#[derive(Debug, Clone)]
pub enum LengthTag {
    VeryShort,
    Short,
    Medium,
    Long,
    VeryLong,
}

impl SpecialTag for LengthTag {
    fn to_tag(&self) -> String {
        match self {
            Self::VeryShort => "<|length:very_short|>".to_string(),
            Self::Short => "<|length:short|>".to_string(),
            Self::Medium => "<|length:medium|>".to_string(),
            Self::Long => "<|length:long|>".to_string(),
            Self::VeryLong => "<|length:very_long|>".to_string(),
        }
    }

    fn is_special(tag: &str) -> bool {
        tag.starts_with("<|length:") && tag.ends_with("|>")
    }
}

impl FromStr for LengthTag {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "very_short" => Ok(Self::VeryShort),
            "short" => Ok(Self::Short),
            "medium" => Ok(Self::Medium),
            "long" => Ok(Self::Long),
            "very_long" => Ok(Self::VeryLong),
            _ => Err(E::msg("invalid length tag")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AspectRatioTag {
    UltraWide,
    Wide,
    Square,
    Tall,
    UltraTall,
}

impl SpecialTag for AspectRatioTag {
    fn to_tag(&self) -> String {
        match self {
            Self::UltraWide => "<|aspect_ratio:ultra_wide|>".to_string(),
            Self::Wide => "<|aspect_ratio:wide|>".to_string(),
            Self::Square => "<|aspect_ratio:square|>".to_string(),
            Self::Tall => "<|aspect_ratio:tall|>".to_string(),
            Self::UltraTall => "<|aspect_ratio:ultra_tall|>".to_string(),
        }
    }

    fn is_special(tag: &str) -> bool {
        tag.starts_with("<|aspect_ratio:") && tag.ends_with("|>")
    }
}

impl FromStr for AspectRatioTag {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ultra_wide" => Ok(Self::UltraWide),
            "wide" => Ok(Self::Wide),
            "square" => Ok(Self::Square),
            "tall" => Ok(Self::Tall),
            "ultra_tall" => Ok(Self::UltraTall),
            _ => Err(E::msg("invalid aspect ratio tag")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RatingTag {
    Sfw,
    General,
    Sensitive,
    Nsfw,
    Questionable,
    Explicit,
}

impl SpecialTag for RatingTag {
    fn to_tag(&self) -> String {
        match self {
            Self::Sfw => "<|rating:sfw|>".to_string(),
            Self::General => "<|rating:general|>".to_string(),
            Self::Sensitive => "<|rating:sensitive|>".to_string(),
            Self::Nsfw => "<|rating:nsfw|>".to_string(),
            Self::Questionable => "<|rating:questionable|>".to_string(),
            Self::Explicit => "<|rating:explicit|>".to_string(),
        }
    }

    fn is_special(tag: &str) -> bool {
        tag.starts_with("<|rating:") && tag.ends_with("|>")
    }
}

impl FromStr for RatingTag {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "sfw" => Ok(Self::Sfw),
            "general" => Ok(Self::General),
            "sensitive" => Ok(Self::Sensitive),
            "nsfw" => Ok(Self::Nsfw),
            "questionable" => Ok(Self::Questionable),
            "explicit" => Ok(Self::Explicit),
            _ => Err(E::msg("invalid rating tag")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum IdentityTag {
    None,
    Lax,
    Strict,
}

impl SpecialTag for IdentityTag {
    fn to_tag(&self) -> String {
        match self {
            Self::None => "<|identity:none|>".to_string(),
            Self::Lax => "<|identity:lax|>".to_string(),
            Self::Strict => "<|identity:strict|>".to_string(),
        }
    }

    fn is_special(tag: &str) -> bool {
        tag.starts_with("<|identity:") && tag.ends_with("|>")
    }
}

impl FromStr for IdentityTag {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "lax" => Ok(Self::Lax),
            "strict" => Ok(Self::Strict),
            _ => Err(E::msg("invalid identity tag")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ReservedTag {
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

impl SpecialTag for ReservedTag {
    fn to_tag(&self) -> String {
        match self {
            Self::Bos => "<|bos|>".to_string(),
            Self::Eos => "<|eos|>".to_string(),
            Self::CopyrightStart => "<copyright>".to_string(),
            Self::CopyrightEnd => "</copyright>".to_string(),
            Self::CharacterStart => "<character>".to_string(),
            Self::CharacterEnd => "</character>".to_string(),
            Self::GeneralStart => "<general>".to_string(),
            Self::GeneralEnd => "</general>".to_string(),
            Self::InputEnd => "<|input_end|>".to_string(),
        }
    }

    fn is_special(tag: &str) -> bool {
        match tag {
            "<|bos|>" => true,
            "<|eos|>" => true,
            "<copyright>" => true,
            "</copyright>" => true,
            "<character>" => true,
            "</character>" => true,
            "<general>" => true,
            "</general>" => true,
            "<|input_end|>" => true,
            _ => false,
        }
    }
}

impl fmt::Display for ReservedTag {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = self.to_tag();
        write!(f, "{}", s)
    }
}

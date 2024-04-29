use anyhow::{Error as E, Result};
use std::fmt;
use std::str::FromStr;

pub trait SpecialTag {
    fn to_tag(&self) -> &str;
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum LengthTag {
    VeryShort,
    Short,
    Medium,
    Long,
    VeryLong,
}

impl SpecialTag for LengthTag {
    fn to_tag(&self) -> &str {
        match self {
            Self::VeryShort => "<|length:very_short|>",
            Self::Short => "<|length:short|>",
            Self::Medium => "<|length:medium|>",
            Self::Long => "<|length:long|>",
            Self::VeryLong => "<|length:very_long|>",
        }
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

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum AspectRatioTag {
    UltraWide,
    Wide,
    Square,
    Tall,
    UltraTall,
}

impl SpecialTag for AspectRatioTag {
    fn to_tag(&self) -> &str {
        match self {
            Self::UltraWide => "<|aspect_ratio:ultra_wide|>",
            Self::Wide => "<|aspect_ratio:wide|>",
            Self::Square => "<|aspect_ratio:square|>",
            Self::Tall => "<|aspect_ratio:tall|>",
            Self::UltraTall => "<|aspect_ratio:ultra_tall|>",
        }
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

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum RatingTag {
    Sfw,
    General,
    Sensitive,
    Nsfw,
    Questionable,
    Explicit,
}

impl SpecialTag for RatingTag {
    fn to_tag(&self) -> &str {
        match self {
            Self::Sfw => "<|rating:sfw|>",
            Self::General => "<|rating:general|>",
            Self::Sensitive => "<|rating:sensitive|>",
            Self::Nsfw => "<|rating:nsfw|>",
            Self::Questionable => "<|rating:questionable|>",
            Self::Explicit => "<|rating:explicit|>",
        }
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

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum IdentityTag {
    None,
    Lax,
    Strict,
}

impl SpecialTag for IdentityTag {
    fn to_tag(&self) -> &str {
        match self {
            Self::None => "<|identity:none|>",
            Self::Lax => "<|identity:lax|>",
            Self::Strict => "<|identity:strict|>",
        }
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
    fn to_tag(&self) -> &str {
        match self {
            Self::Bos => "<|bos|>",
            Self::Eos => "<|eos|>",
            Self::CopyrightStart => "<copyright>",
            Self::CopyrightEnd => "</copyright>",
            Self::CharacterStart => "<character>",
            Self::CharacterEnd => "</character>",
            Self::GeneralStart => "<general>",
            Self::GeneralEnd => "</general>",
            Self::InputEnd => "<|input_end|>",
        }
    }
}

impl fmt::Display for ReservedTag {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = self.to_tag();
        write!(f, "{}", s)
    }
}

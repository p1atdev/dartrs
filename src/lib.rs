pub mod bindings;
pub mod configs;
pub mod generate;
pub mod models;
pub mod polyfill;
pub mod prompt;
pub mod tags;

use generate::CausalLM;
use models::*;

pub enum ModelWraapper {
    Mistral(MistralModelFamily),
    Mixtral(MixtralModelFamily),
    // Llama(LlamaModelFamily),
}

impl ModelWraapper {}

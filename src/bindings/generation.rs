use crate::bindings::models::{DartDevice, DartTokenizer};
use crate::generation::GenerationConfig;

use candle_core::Device;
use pyo3::prelude::*;
use tokenizers::Tokenizer;

#[pyclass(name = "GenerationConfig")]
#[derive(Clone)]
pub struct DartGenerationConfig {
    device: DartDevice,
    tokenizer: DartTokenizer,
    prompt: String,
    eos_token: Option<u32>,
    max_new_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    seed: Option<u64>,
}

impl From<DartGenerationConfig> for GenerationConfig {
    fn from(config: DartGenerationConfig) -> Self {
        GenerationConfig::new(
            Device::from(config.device),
            Tokenizer::from(config.tokenizer),
            config.prompt,
            config.eos_token,
            config.max_new_tokens,
            config.temperature,
            config.top_p,
            config.top_k,
            config.seed,
        )
    }
}

#[pymethods]
impl DartGenerationConfig {
    #[new]
    fn new(
        device: DartDevice,
        tokenizer: DartTokenizer,
        prompt: String,
        eos_token: Option<u32>,
        max_new_tokens: Option<usize>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            device,
            tokenizer,
            prompt,
            eos_token,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            seed,
        }
    }
}

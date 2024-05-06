use crate::bindings::models::{DartDevice, DartTokenizer};
use crate::generation::{GenerationCache, GenerationConfig};

use candle_core::Device;
use pyo3::prelude::*;
use tokenizers::Tokenizer;

#[pyclass(name = "GenerationConfig")]
#[derive(Clone, Debug)]
pub(crate) struct DartGenerationConfig {
    device: DartDevice,
    tokenizer: DartTokenizer,
    prompt: String,
    eos_token: Option<u32>,
    max_new_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    ban_token_ids: Option<Vec<u32>>,
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
            config.ban_token_ids,
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
        ban_token_ids: Option<Vec<u32>>,
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
            ban_token_ids,
            seed,
        }
    }

    fn tokenizer(&self) -> DartTokenizer {
        self.tokenizer.clone()
    }

    fn prompt(&self) -> &str {
        &self.prompt
    }

    fn max_new_tokens(&self) -> Option<usize> {
        self.max_new_tokens
    }
}

#[pyclass(name = "GenerationCache")]
#[derive(Clone, Debug)]
pub(crate) struct DartGenerationCache {
    pub input_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub finished: bool,
}

impl From<DartGenerationCache> for GenerationCache {
    fn from(cache: DartGenerationCache) -> Self {
        GenerationCache {
            input_tokens: cache.input_tokens,
            output_tokens: cache.output_tokens,
            finished: cache.finished,
        }
    }
}

impl From<GenerationCache> for DartGenerationCache {
    fn from(cache: GenerationCache) -> Self {
        DartGenerationCache {
            input_tokens: cache.input_tokens,
            output_tokens: cache.output_tokens,
            finished: cache.finished,
        }
    }
}

#[pymethods]
impl DartGenerationCache {
    #[new]
    fn new(input_tokens: Vec<u32>) -> Self {
        Self {
            input_tokens,
            output_tokens: Vec::new(),
            finished: false,
        }
    }

    fn clear(&mut self) {
        self.output_tokens.clear();
        self.finished = false;
    }

    fn input_tokens(&self) -> Vec<u32> {
        self.input_tokens.clone()
    }

    fn output_tokens(&self) -> Vec<u32> {
        self.output_tokens.clone()
    }

    fn finished(&self) -> bool {
        self.finished
    }
}

use std::collections::HashMap;

use crate::bindings::generation::{DartGenerationCache, DartGenerationConfig};
use crate::generation::{GenerationCache, GenerationConfig, TextGeneration};
use crate::models::{
    mistral, mixtral, MistralModelBuilder, MixtralModelBuilder, ModelBuilder, ModelRepositoy,
};

use candle_core::{DType, Device};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Repo;
use hf_hub::RepoType;
use tokenizers::Tokenizer;

use pyo3::exceptions;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub(crate) enum DartDType {
    BF16,
    FP16,
    FP32,
}

impl From<DartDType> for DType {
    fn from(dtype: DartDType) -> Self {
        match dtype {
            DartDType::BF16 => DType::BF16,
            DartDType::FP16 => DType::F16,
            DartDType::FP32 => DType::F32,
        }
    }
}

#[pymethods]
impl DartDType {
    #[new]
    fn from(dtype: String) -> PyResult<Self> {
        match dtype.as_str() {
            "bf16" => Ok(DartDType::BF16),
            "fp16" => Ok(DartDType::FP16),
            "fp32" => Ok(DartDType::FP32),
            _ => Err(exceptions::PyValueError::new_err("invalid dtype")),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub(crate) enum DartDevice {
    Cpu {},
    Cuda { id: usize },
}

impl From<DartDevice> for Device {
    fn from(device: DartDevice) -> Self {
        match device {
            DartDevice::Cpu {} => Device::Cpu,
            DartDevice::Cuda { id } => match Device::cuda_if_available(id) {
                Ok(device) => device,
                Err(_e) => Device::Cpu,
            },
        }
    }
}

#[pymethods]
impl DartDevice {
    #[new]
    fn new(device: String) -> PyResult<Self> {
        if device.starts_with("cuda") {
            let id = device[4..].parse().unwrap_or(0);
            Ok(DartDevice::Cuda { id })
        } else if device == "cpu" {
            Ok(DartDevice::Cpu {})
        } else {
            Err(exceptions::PyValueError::new_err("invalid device"))
        }
    }
}

macro_rules! generate {
    ($self:ident, $config:ident) => {
        match $self.model.generate(&mut $config) {
            Ok(text) => Ok(text),
            Err(e) => Err(exceptions::PyOSError::new_err(format!(
                "Failed to generate text: {}",
                e
            ))),
        }
    };
}

#[pyclass]
pub(crate) struct DartV2Mistral {
    model: mistral::Model,
}

impl From<mistral::Model> for DartV2Mistral {
    fn from(model: mistral::Model) -> Self {
        Self { model }
    }
}

#[pymethods]
impl DartV2Mistral {
    #[new]
    fn new(
        hub_name: String,
        revision: Option<String>,
        dtype: Option<DartDType>,
        device: Option<DartDevice>,
        auth_token: Option<String>,
    ) -> PyResult<Self> {
        let builder = ApiBuilder::default();
        let builder = builder.with_token(auth_token);
        let api = match builder.build() {
            Ok(api) => api,
            Err(e) => {
                return Err(exceptions::PyOSError::new_err(format!(
                    "Failed to create API: {}",
                    e
                )))
            }
        };
        let dtype = dtype.unwrap_or(DartDType::FP32);
        let dtype = DType::from(dtype);
        let device = device.unwrap_or(DartDevice::Cpu {});
        let device = Device::from(device);

        let repo = ModelRepositoy::new(hub_name.clone(), api.clone(), revision);

        let model = MistralModelBuilder::load(&repo, dtype, &device);
        match model {
            Ok(model) => Ok(Self { model }),
            Err(e) => Err(exceptions::PyOSError::new_err(format!(
                "Failed to load model: {}",
                e
            ))),
        }
    }

    fn generate(&mut self, config: DartGenerationConfig) -> PyResult<String> {
        let mut config = GenerationConfig::from(config);
        generate!(self, config)
    }

    fn get_next_token(
        &mut self,
        config: DartGenerationConfig,
        cache: DartGenerationCache,
    ) -> PyResult<(u32, DartGenerationCache)> {
        let mut config = GenerationConfig::from(config);
        let mut cache = GenerationCache::from(cache);
        match self.model.get_next_token(&mut config, &mut cache) {
            Ok(token) => Ok((token, DartGenerationCache::from(cache))),
            Err(e) => Err(exceptions::PyOSError::new_err(format!(
                "Failed to get next token: {}",
                e
            ))),
        }
    }

    fn _clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

#[pyclass]
pub(crate) struct DartV2Mixtral {
    model: mixtral::Model,
}

impl From<mixtral::Model> for DartV2Mixtral {
    fn from(model: mixtral::Model) -> Self {
        Self { model }
    }
}

#[pymethods]
impl DartV2Mixtral {
    #[new]
    fn new(
        hub_name: String,
        revision: Option<String>,
        dtype: Option<DartDType>,
        device: Option<DartDevice>,
        auth_token: Option<String>,
    ) -> PyResult<Self> {
        let builder = ApiBuilder::default();
        let builder = builder.with_token(auth_token);
        let api = match builder.build() {
            Ok(api) => api,
            Err(e) => {
                return Err(exceptions::PyOSError::new_err(format!(
                    "Failed to create API: {}",
                    e
                )))
            }
        };
        let dtype = dtype.unwrap_or(DartDType::FP32);
        let device = device.unwrap_or(DartDevice::Cpu {});
        let device = Device::from(device);
        let dtype = DType::from(dtype);

        let repo = ModelRepositoy::new(hub_name.clone(), api.clone(), revision);

        let model = MixtralModelBuilder::load(&repo, dtype, &device);
        match model {
            Ok(model) => Ok(Self { model }),
            Err(e) => Err(exceptions::PyOSError::new_err(format!(
                "Failed to load model: {}",
                e
            ))),
        }
    }

    fn generate(&mut self, config: DartGenerationConfig) -> PyResult<String> {
        let mut config = GenerationConfig::from(config);
        generate!(self, config)
    }

    fn get_next_token(
        &mut self,
        config: DartGenerationConfig,
        cache: DartGenerationCache,
    ) -> PyResult<(u32, DartGenerationCache)> {
        let mut config = GenerationConfig::from(config);
        let mut cache = GenerationCache::from(cache);
        match self.model.get_next_token(&mut config, &mut cache) {
            Ok(token) => Ok((token, DartGenerationCache::from(cache))),
            Err(e) => Err(exceptions::PyOSError::new_err(format!(
                "Failed to get next token: {}",
                e
            ))),
        }
    }

    fn _clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct DartTokenizer {
    pub tokenizer: Tokenizer,
}

impl DartTokenizer {
    fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }
}

impl From<DartTokenizer> for Tokenizer {
    fn from(tokenizer: DartTokenizer) -> Self {
        tokenizer.tokenizer
    }
}

#[pymethods]
impl DartTokenizer {
    #[staticmethod]
    #[pyo3(signature = (identifier, revision = String::from("main"), auth_token = None))]
    fn from_pretrained(
        identifier: &str,
        revision: String,
        auth_token: Option<String>,
    ) -> PyResult<Self> {
        let builder = ApiBuilder::default();
        let builder = builder.with_token(auth_token);
        let api = match builder.build() {
            Ok(api) => api,
            Err(e) => {
                return Err(exceptions::PyOSError::new_err(format!(
                    "Failed to create API: {}",
                    e
                )))
            }
        };
        let repo = api.repo(Repo::with_revision(
            identifier.to_string(),
            RepoType::Model,
            revision,
        ));
        let tokenizer_json = match repo.get("tokenizer.json") {
            Ok(tokenizer_json) => tokenizer_json,
            Err(e) => {
                return Err(exceptions::PyOSError::new_err(format!(
                    "Failed to get tokenizer.json: {}",
                    e
                )))
            }
        };
        let tokenizer = Tokenizer::from_file(tokenizer_json).map_err(|e| {
            exceptions::PyOSError::new_err(format!("Failed to load tokenizer: {}", e))
        })?;

        Ok(Self::new(tokenizer))
    }

    fn encode(&self, text: String) -> PyResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false) // add_special_tokens = false
            .map_err(|e| exceptions::PyOSError::new_err(format!("Failed to encode text: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode_tags(
        &self,
        tokens: Vec<u32>,
        skip_special_tokens: Option<bool>,
    ) -> PyResult<Vec<String>> {
        let skip_special_tokens = skip_special_tokens.unwrap_or(true);
        let tags = tokens
            .iter()
            .map(|&token| {
                self.tokenizer
                    .decode(&[token], skip_special_tokens)
                    .unwrap()
            })
            .filter(|tag| !tag.is_empty())
            .collect::<Vec<_>>();
        Ok(tags)
    }

    fn decode(&self, tokens: Vec<u32>, skip_special_tokens: Option<bool>) -> PyResult<String> {
        let tags = self.decode_tags(tokens, skip_special_tokens)?;
        let decoded = tags.join(", ");
        Ok(decoded)
    }

    fn tokenize(&self, text: String) -> PyResult<Vec<String>> {
        let tokens = self.tokenizer.encode(text, false).map_err(|e| {
            exceptions::PyOSError::new_err(format!("Failed to tokenize text: {}", e))
        })?;

        Ok(tokens.get_tokens().to_vec())
    }

    fn get_vocab(&self, with_added_tokens: Option<bool>) -> HashMap<String, u32> {
        let with_added_tokens = with_added_tokens.unwrap_or(true);
        self.tokenizer.get_vocab(with_added_tokens)
    }

    fn get_added_tokens(&self) -> Vec<String> {
        let added_tokens = self.tokenizer.get_added_tokens_decoder();
        added_tokens
            .values()
            .cloned()
            .map(|token| token.content.to_string())
            .collect()
    }
}

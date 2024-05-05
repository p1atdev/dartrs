use crate::bindings::generation::DartGenerationConfig;
use crate::generation::{GenerationConfig, TextGeneration};
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

#[pyclass]
pub(crate) struct DartV2Mistral(mistral::Model);

impl From<mistral::Model> for DartV2Mistral {
    fn from(model: mistral::Model) -> Self {
        Self(model)
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
        let device = device.unwrap_or(DartDevice::Cpu {});
        let device = Device::from(device);
        let dtype = DType::from(dtype);

        let repo = ModelRepositoy::new(hub_name.clone(), api.clone(), revision);

        let model = MistralModelBuilder::load(&repo, dtype, &device);
        match model {
            Ok(model) => Ok(Self(model)),
            Err(e) => Err(exceptions::PyOSError::new_err(format!(
                "Failed to load model: {}",
                e
            ))),
        }
    }

    fn generate(&mut self, config: DartGenerationConfig) -> PyResult<String> {
        let mut config = GenerationConfig::from(config);
        match self.0.generate(&mut config) {
            Ok(text) => Ok(text),
            Err(e) => Err(exceptions::PyOSError::new_err(format!(
                "Failed to generate text: {}",
                e
            ))),
        }
    }
}

#[pyclass]
pub(crate) struct DartV2Mixtral(mixtral::Model);

impl From<mixtral::Model> for DartV2Mixtral {
    fn from(model: mixtral::Model) -> Self {
        Self(model)
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
            Ok(model) => Ok(Self(model)),
            Err(e) => Err(exceptions::PyOSError::new_err(format!(
                "Failed to load model: {}",
                e
            ))),
        }
    }

    fn generate(&mut self, config: DartGenerationConfig) -> PyResult<String> {
        let mut config = GenerationConfig::from(config);
        match self.0.generate(&mut config) {
            Ok(text) => Ok(text),
            Err(e) => Err(exceptions::PyOSError::new_err(format!(
                "Failed to generate text: {}",
                e
            ))),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct DartTokenizer(Tokenizer);

impl DartTokenizer {
    fn new(tokenizer: Tokenizer) -> Self {
        Self(tokenizer)
    }
}

impl From<DartTokenizer> for Tokenizer {
    fn from(tokenizer: DartTokenizer) -> Self {
        tokenizer.0
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
            .0
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
            .map(|&token| self.0.decode(&[token], skip_special_tokens).unwrap())
            .filter(|tag| !tag.is_empty())
            .collect::<Vec<_>>();
        Ok(tags)
    }

    fn decode(&self, tokens: Vec<u32>, skip_special_tokens: Option<bool>) -> PyResult<String> {
        let tags = self.decode_tags(tokens, skip_special_tokens)?;
        let decoded = tags.join(", ");
        Ok(decoded)
    }
}

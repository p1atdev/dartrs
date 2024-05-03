use crate::models::MistralModelBuilder;
use crate::models::MixtralModelBuilder;
use crate::models::ModelBuilder;

use candle_core::{DType, Device};
use candle_transformers::models::{mistral, mixtral};
use hf_hub::api::sync::ApiBuilder;
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
pub(crate) struct DartMistral(mistral::Model);

impl From<mistral::Model> for DartMistral {
    fn from(model: mistral::Model) -> Self {
        Self(model)
    }
}

impl DartMistral {
    fn load_v2(
        hub_name: String,
        revision: Option<String>,
        dtype: Option<DartDType>,
        device: Option<DartDevice>,
    ) -> PyResult<Self> {
        let api = match ApiBuilder::default().build() {
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

        let model = MistralModelBuilder::load(hub_name, &api, dtype, &device);
        match model {
            Ok(model) => Ok(Self(model)),
            Err(e) => Err(exceptions::PyOSError::new_err(format!(
                "Failed to load model: {}",
                e
            ))),
        }
    }
}

#[pyclass]
pub(crate) struct DartMixtral(mixtral::Model);

impl From<mixtral::Model> for DartMixtral {
    fn from(model: mixtral::Model) -> Self {
        Self(model)
    }
}

impl DartMixtral {
    fn load_v2(
        hub_name: String,
        revision: Option<String>,
        dtype: Option<DartDType>,
        device: Option<DartDevice>,
    ) -> PyResult<Self> {
        let api = match ApiBuilder::default().build() {
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

        let model = MixtralModelBuilder::load(hub_name, &api, dtype, &device);
        match model {
            Ok(model) => Ok(Self(model)),
            Err(e) => Err(exceptions::PyOSError::new_err(format!(
                "Failed to load model: {}",
                e
            ))),
        }
    }
}

#[pyclass]
pub(crate) struct DartTokenizer(Tokenizer);

impl From<Tokenizer> for DartTokenizer {
    fn from(tokenizer: Tokenizer) -> Self {
        Self(tokenizer)
    }
}

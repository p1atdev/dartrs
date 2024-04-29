use anyhow::{Error as E, Result};

use crate::configs::{DartV2Mistral, DartV2Mixtral};
use candle_transformers::models::{mistral, mixtral};
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub trait ModelBuilder<T> {
    fn build(&self) -> Result<T>;
}

pub struct MistralModelBuilder<T> {
    repo: ApiRepo,
    dtype: DType,
    device: Device,
    config: T,
}

impl ModelBuilder<mistral::Model> for MistralModelBuilder<mistral::Config> {
    fn build(&self) -> Result<mistral::Model> {
        let model_path = self.repo.get("model.safetensors")?;
        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], self.dtype, &self.device)?
        };
        let model = mistral::Model::new(&self.config, var_builder)?;
        Ok(model)
    }
}

impl MistralModelBuilder<mistral::Config> {
    pub fn dart_v2_100m(api: &Api, dtype: DType, device: &Device) -> Self {
        let repo = api.repo(Repo::with_revision(
            "p1atdev/dart-v2-mistral-100m-sft".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let config = mistral::Config::dart_v2_100m();
        Self {
            repo,
            dtype,
            device: device.clone(),
            config,
        }
    }
}

pub struct MixtralModelBuilder<T> {
    repo: ApiRepo,
    dtype: DType,
    device: Device,
    config: T,
}

impl ModelBuilder<mixtral::Model> for MixtralModelBuilder<mixtral::Config> {
    fn build(&self) -> Result<mixtral::Model> {
        let model_path = self.repo.get("model.safetensors")?;
        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], self.dtype, &self.device)?
        };
        let model = mixtral::Model::new(&self.config, var_builder)?;
        Ok(model)
    }
}

impl MixtralModelBuilder<mixtral::Config> {
    pub fn dart_v2_160m(api: &Api, dtype: DType, device: &Device) -> Self {
        let repo = api.repo(Repo::with_revision(
            "p1atdev/dart-v2-mixtral-160m-sft".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let config = mixtral::Config::dart_v2_160m();
        Self {
            repo,
            dtype,
            device: device.clone(),
            config,
        }
    }
}

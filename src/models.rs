use anyhow::Result;

use crate::configs::*;
use crate::polyfill::*;
use candle_transformers::models::{llama, mistral, mixtral};
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub trait ModelBuilder<T> {
    fn build(&self) -> Result<T>;
    fn new(model_name: String, api: &Api, dtype: DType, device: &Device) -> Self;
}

pub trait ModelFamily<T> {
    fn load(&self, api: &Api, dtype: DType, device: &Device) -> Result<T>;
}

pub enum ModelRepositoy {
    V2Llama100m,
    V2Mistral100m,
    V2Mixtral160m,
}

impl ModelRepositoy {
    pub fn hub_name(&self) -> String {
        match self {
            ModelRepositoy::V2Llama100m => "p1atdev/dart-v2-llama-100m-sft".to_string(),
            ModelRepositoy::V2Mistral100m => "p1atdev/dart-v2-mistral-100m-sft".to_string(),
            ModelRepositoy::V2Mixtral160m => "p1atdev/dart-v2-mixtral-160m-sft".to_string(),
        }
    }
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

    fn new(model_name: String, api: &Api, dtype: DType, device: &Device) -> Self {
        let repo = api.repo(Repo::with_revision(
            model_name,
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

pub enum MistralModelFamily {
    V2_100m,
}

impl ModelFamily<mistral::Model> for MistralModelFamily {
    fn load(&self, api: &Api, dtype: DType, device: &Device) -> Result<mistral::Model> {
        let model_name = match self {
            MistralModelFamily::V2_100m => ModelRepositoy::V2Mistral100m.hub_name().to_string(),
        };
        let builder = MistralModelBuilder::new(model_name, api, dtype, device);
        builder.build()
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

    fn new(model_name: String, api: &Api, dtype: DType, device: &Device) -> Self {
        let repo = api.repo(Repo::with_revision(
            model_name,
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

pub enum MixtralModelFamily {
    V2_160m,
}

impl ModelFamily<mixtral::Model> for MixtralModelFamily {
    fn load(&self, api: &Api, dtype: DType, device: &Device) -> Result<mixtral::Model> {
        let model_name = match self {
            MixtralModelFamily::V2_160m => ModelRepositoy::V2Mixtral160m.hub_name().to_string(),
        };
        let builder = MixtralModelBuilder::new(model_name, api, dtype, device);
        builder.build()
    }
}

pub struct LlamaModelBuilder<T> {
    repo: ApiRepo,
    dtype: DType,
    device: Device,
    config: T,
}

impl ModelBuilder<llama::Llama> for LlamaModelBuilder<llama::Config> {
    fn build(&self) -> Result<llama::Llama> {
        let model_path = self.repo.get("model.safetensors")?;
        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], self.dtype, &self.device)?
        };
        let model = llama::Llama::new(&self.config, var_builder)?;
        Ok(model)
    }

    fn new(model_name: String, api: &Api, dtype: DType, device: &Device) -> Self {
        let repo = api.repo(Repo::with_revision(
            model_name,
            RepoType::Model,
            "main".to_string(),
        ));

        let config = llama::Config::dart_v2_100m();
        Self {
            repo,
            dtype,
            device: device.clone(),
            config,
        }
    }
}

pub enum LlamaModelFamily {
    V2_100m,
}

impl ModelFamily<llama::Llama> for LlamaModelFamily {
    fn load(&self, api: &Api, dtype: DType, device: &Device) -> Result<llama::Llama> {
        let model_name = match self {
            LlamaModelFamily::V2_100m => ModelRepositoy::V2Llama100m.hub_name().to_string(),
        };
        let builder = LlamaModelBuilder::new(model_name, api, dtype, device);
        builder.build()
    }
}

pub mod mistral;
pub mod mixtral;

use anyhow::{Error as E, Result};

use crate::configs::*;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

pub trait ModelBuilder<T> {
    fn build(&self) -> Result<T>;
    fn new(repo: &ModelRepositoy, dtype: DType, device: &Device) -> Self;
    fn load(repo: &ModelRepositoy, dtype: DType, device: &Device) -> Result<T>;
}

pub struct ModelRepositoy {
    hub_name: String,
    api: Api,
    revision: String,
}

impl ModelRepositoy {
    pub fn new(hub_name: String, api: Api, revision: Option<String>) -> Self {
        Self {
            hub_name,
            api,
            revision: revision.unwrap_or("main".to_string()),
        }
    }

    pub fn hub_name(&self) -> String {
        self.hub_name.clone()
    }

    pub fn load_tokenizer(&self) -> Result<Tokenizer> {
        let repo = &self.api.repo(Repo::with_revision(
            self.hub_name().clone(),
            RepoType::Model,
            self.revision.clone(),
        ));
        let tokenizer_json = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_json).map_err(E::msg)?;
        Ok(tokenizer)
    }

    fn api_repo(&self) -> ApiRepo {
        self.api.repo(Repo::with_revision(
            self.hub_name.clone(),
            RepoType::Model,
            self.revision.clone(),
        ))
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

    fn new(repo: &ModelRepositoy, dtype: DType, device: &Device) -> Self {
        let config = mistral::Config::v2_100m(false);
        Self {
            repo: repo.api_repo(),
            dtype,
            device: device.clone(),
            config,
        }
    }

    fn load(repo: &ModelRepositoy, dtype: DType, device: &Device) -> Result<mistral::Model> {
        let builder = MistralModelBuilder::new(repo, dtype, device);
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

    fn new(repo: &ModelRepositoy, dtype: DType, device: &Device) -> Self {
        let config = mixtral::Config::v2_160m(false);
        Self {
            repo: repo.api_repo(),
            dtype,
            device: device.clone(),
            config,
        }
    }

    fn load(repo: &ModelRepositoy, dtype: DType, device: &Device) -> Result<mixtral::Model> {
        let builder = MixtralModelBuilder::new(repo, dtype, device);
        builder.build()
    }
}

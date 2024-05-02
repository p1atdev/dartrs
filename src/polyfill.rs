use anyhow::{Error as E, Result};

use candle_nn::VarBuilder;
use candle_transformers::models::llama;

pub trait CustomLlama {
    fn new(cfg: &llama::Config, vb: VarBuilder) -> Result<llama::Llama>;
}

impl CustomLlama for llama::Llama {
    fn new(cfg: &llama::Config, vb: VarBuilder) -> Result<llama::Llama> {
        match llama::Llama::load(vb, &cfg) {
            Ok(llama) => Ok(llama),
            Err(err) => Err(E::msg(err)),
        }
    }
}

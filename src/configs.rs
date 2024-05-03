use candle_nn::Activation;

use crate::models::{mistral, mixtral};

pub trait DartV2Mistral {
    fn v2_100m(use_flash_attn: bool) -> Self;
}

impl DartV2Mistral for mistral::Config {
    fn v2_100m(use_flash_attn: bool) -> Self {
        Self {
            vocab_size: 30649,
            hidden_act: Activation::Silu,
            hidden_size: 768,
            intermediate_size: 3072,
            max_position_embeddings: 1024,
            num_attention_heads: 8,
            num_hidden_layers: 8,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-05,
            rope_theta: 1000.0,
            sliding_window: None,
            use_flash_attn,
        }
    }
}

pub trait DartV2Mixtral {
    fn v2_160m(use_flash_attn: bool) -> Self;
}

impl DartV2Mixtral for mixtral::Config {
    fn v2_160m(use_flash_attn: bool) -> Self {
        Self {
            vocab_size: 30649,
            hidden_act: Activation::Silu,
            hidden_size: 768,
            intermediate_size: 3072,
            max_position_embeddings: 1024,
            num_attention_heads: 8,
            num_experts_per_tok: 2,
            num_hidden_layers: 4,
            num_key_value_heads: 1,
            num_local_experts: 4,
            rms_norm_eps: 1e-05,
            rope_theta: 1000.0,
            sliding_window: None,
            use_flash_attn,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_100m() {
        mistral::Config::v2_100m(false);
    }

    #[test]
    fn test_mixtral_160m() {
        mixtral::Config::v2_160m(false);
    }
}

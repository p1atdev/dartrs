use std::panic;

use candle_nn::Activation;
use candle_transformers::models::{mistral, mixtral};
use serde_json::from_str;

pub trait DartV2Mistral {
    fn dart_v2_100m() -> Self;
}

impl DartV2Mistral for mistral::Config {
    fn dart_v2_100m() -> Self {
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
            use_flash_attn: false,
        }
    }
}

pub trait DartV2Mixtral {
    fn dart_v2_160m() -> Self;
}

impl DartV2Mixtral for mixtral::Config {
    // currenty the mixtral::Config of candle_transformer is not accessible
    // so we will use serde_json to parse the config
    fn dart_v2_160m() -> Self {
        if let Ok(config) = from_str(
            r#"
        {
            "vocab_size": 30649,
            "hidden_act": "silu",
            "hidden_size": 768,
            "intermediate_size": 3072,
            "max_position_embeddings": 1024,
            "num_attention_heads": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "num_key_value_heads": 1,
            "num_local_experts": 4,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000.0,
            "sliding_window": 0,
            "use_flash_attn": false
        }
        "#,
        ) {
            config
        } else {
            panic!("failed to parse config")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixtral_160m() {
        mixtral::Config::dart_v2_160m();
    }
}

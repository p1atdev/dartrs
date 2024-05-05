use candle_core::{DType, Error, Result, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};

pub struct DartLogitsProcessor {
    logits_processor: LogitsProcessor,
    // 出現を禁止するトークンのID (つまり、インデックス) の配列
    ban_token_ids: Vec<u32>,
}

pub fn ban_tokens(prs: &mut [f32], ban_token_ids: Vec<u32>) {
    // トークンのインデックスにある確率を 0 にする
    for token_id in ban_token_ids {
        prs[token_id as usize] = 0.0;
    }
}

impl DartLogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling, ban_token_ids: Option<Vec<u32>>) -> Self {
        let ban_token_ids = match ban_token_ids {
            Some(ban_token_ids) => ban_token_ids,
            None => Vec::new(),
        };
        let logits_processor = LogitsProcessor::from_sampling(seed, sampling);
        Self {
            logits_processor,
            ban_token_ids,
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        self.logits_processor
            .sample_f(logits, |prs| ban_tokens(prs, self.ban_token_ids.clone()))
    }
}

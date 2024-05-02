use anyhow::{Error as E, Result};

use candle_transformers::{
    generation::Sampling,
    models::{mistral, mixtral},
};

use candle_core::{DType, Device, Tensor};

use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

pub trait CausalLM {
    fn common_forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>;
}

pub struct GenerationCache {
    input_tokens: Vec<u32>,
    output_tokens: Vec<u32>,
    finished: bool,
}

impl GenerationCache {
    fn new(tokens: Option<Vec<u32>>) -> Self {
        let tokens = match tokens {
            Some(tokens) => tokens,
            None => Vec::new(),
        };

        Self {
            input_tokens: tokens,
            output_tokens: Vec::new(),
            finished: false,
        }
    }
}

pub struct TextGeneration<T>
where
    T: CausalLM,
{
    model: T,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    eos_token: u32,
}

impl<T> TextGeneration<T>
where
    T: CausalLM,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: T,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        device: &Device,
        eos_token: Option<&str>,
    ) -> Self {
        let logits_processor = LogitsProcessor::from_sampling(
            seed,
            Sampling::TopKThenTopP {
                k: top_k.unwrap_or(100),
                p: top_p.unwrap_or(0.9),
                temperature: temp.unwrap_or(1.0),
            },
        );
        let eos_token = eos_token.unwrap_or_else(|| "<|eos|>");
        let eos_token = match &tokenizer.get_vocab(true).get(eos_token).copied() {
            Some(token) => token.clone(),
            None => panic!("cannot find the <|eos|> token"),
        };

        Self {
            model,
            tokenizer,
            logits_processor,
            device: device.clone(),
            eos_token: eos_token,
        }
    }

    fn get_next_token(&mut self, cache: &mut GenerationCache) -> Result<u32> {
        // skip the last token due to the input_end token
        let context_size = if cache.output_tokens.is_empty() {
            cache.input_tokens.len()
        } else {
            1
        };
        // tokens = input_tokens + output_tokens
        let tokens: Vec<u32> = cache
            .input_tokens
            .iter()
            .chain(cache.output_tokens.iter())
            .cloned()
            .collect();

        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
        let logits = self.model.common_forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

        let next_token = self.logits_processor.sample(&logits)?;
        cache.output_tokens.push(next_token);

        if next_token == self.eos_token {
            cache.finished = true;
        }

        Ok(next_token)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        if let Ok(text) = self.tokenizer.decode(&tokens, false) {
            Ok(text)
        } else {
            Err(E::msg("Error in decoding"))
        }
    }

    pub fn run(&mut self, prompt: &str, max_new_tokens: usize) -> Result<()> {
        use std::io::Write;

        let tokens = self
            .tokenizer
            .encode(prompt, false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // print input prompt
        for &t in tokens.iter() {
            if let Ok(t) = self.tokenizer.decode(&[t], false) {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let start_gen = std::time::Instant::now();
        // sampling
        let mut cache = GenerationCache::new(Some(tokens));
        for _ in 0..max_new_tokens {
            let token = self.get_next_token(&mut cache)?;
            if let Ok(tag) = self.decode(&[token]) {
                print!("{tag}, ");
            }

            if cache.finished {
                break;
            }
        }
        let dt = start_gen.elapsed(); // finish

        let generated_tokens = cache.output_tokens.len();

        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }

    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        let tokens = self
            .tokenizer
            .encode(prompt, false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // sampling
        let mut cache = GenerationCache::new(Some(tokens));
        for _ in 0..max_new_tokens {
            self.get_next_token(&mut cache)?;

            if cache.finished {
                break;
            }
        }

        // decode the tokens
        let decoded = self.decode(&cache.output_tokens)?;

        Ok(decoded)
    }
}

// impl CausalLM for llama::Llama {
//     fn common_forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
//         if let Ok(output) = self.forward(input_ids, seqlen_offset) {
//             Ok(output)
//         } else {
//             Err(E::msg("Error in forward"))
//         }
//     }
// }

impl CausalLM for mixtral::Model {
    fn common_forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        if let Ok(output) = self.forward(input_ids, seqlen_offset) {
            Ok(output)
        } else {
            Err(E::msg("Error in forward"))
        }
    }
}

impl CausalLM for mistral::Model {
    fn common_forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        if let Ok(output) = self.forward(input_ids, seqlen_offset) {
            Ok(output)
        } else {
            Err(E::msg("Error in forward"))
        }
    }
}

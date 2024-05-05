use anyhow::{Error as E, Result};

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::Sampling;
use rand::Rng;
use tokenizers::Tokenizer;

use crate::logits_processor::DartLogitsProcessor;
use crate::models::{mistral, mixtral};
use crate::tags::ReservedTag;
use crate::tags::SpecialTag;

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

pub struct GenerationConfig {
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: DartLogitsProcessor,
    eos_token: u32,
    max_new_tokens: usize,
    prompt: String,
}

impl GenerationConfig {
    pub fn new(
        device: Device,
        tokenizer: Tokenizer,
        prompt: String,
        eos_token: Option<u32>,
        max_new_tokens: Option<usize>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        ban_token_ids: Option<Vec<u32>>,
        seed: Option<u64>,
    ) -> Self {
        let sampling = match top_k {
            Some(k) => match top_p {
                Some(p) => match temperature {
                    Some(temperature) => Sampling::TopKThenTopP { k, p, temperature },
                    None => Sampling::TopK {
                        k,
                        temperature: 1.0,
                    },
                },
                None => Sampling::TopK {
                    k,
                    temperature: 1.0,
                },
            },
            None => match top_p {
                Some(p) => match temperature {
                    Some(temperature) => Sampling::TopP { p, temperature },
                    None => Sampling::TopP {
                        p,
                        temperature: 1.0,
                    },
                },
                None => match temperature {
                    Some(temperature) => Sampling::All { temperature },
                    None => Sampling::ArgMax,
                },
            },
        };
        let seed = match seed {
            Some(seed) => seed,
            None => {
                let mut rng = rand::thread_rng();
                rng.gen()
            }
        };
        let logits_processor = DartLogitsProcessor::from_sampling(seed, sampling, ban_token_ids);

        let eos_token = eos_token.unwrap_or_else(|| tokenizer.token_to_id("<|eos|>").unwrap());
        let max_new_tokens = max_new_tokens.unwrap_or(256);

        Self {
            device,
            tokenizer,
            logits_processor,
            eos_token: eos_token.clone(),
            max_new_tokens,
            prompt,
        }
    }

    pub fn default(device: Device, tokenizer: Tokenizer, prompt: String) -> Self {
        Self::new(
            device, tokenizer, prompt, None, None, None, None, None, None, None,
        )
    }
}

pub trait TextGeneration {
    fn get_next_token(
        &mut self,
        config: &mut GenerationConfig,
        cache: &mut GenerationCache,
    ) -> Result<u32>;

    fn decode(&self, config: &mut GenerationConfig, tokens: &[u32]) -> Result<String>;

    fn generate_tokens(&mut self, config: &mut GenerationConfig) -> Result<Vec<String>>;
    fn generate(&mut self, config: &mut GenerationConfig) -> Result<String> {
        let tokens = self.generate_tokens(config)?;

        let text = tokens
            .into_iter()
            .filter(|token| !ReservedTag::is_special(token))
            .collect::<Vec<String>>()
            .join(", ");

        Ok(text)
    }

    fn run(&mut self, config: &mut GenerationConfig) -> Result<()> {
        use std::io::Write;

        let tokens = config
            .tokenizer
            .encode(config.prompt.clone(), false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // print input prompt
        for &t in tokens.iter() {
            if let Ok(t) = config.tokenizer.decode(&[t], false) {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let start_gen = std::time::Instant::now();
        // sampling
        let mut cache = GenerationCache::new(Some(tokens));
        for _ in 0..config.max_new_tokens {
            let token = self.get_next_token(config, &mut cache)?;
            if let Ok(tag) = self.decode(config, &[token]) {
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
}

macro_rules! get_next_token {
    ($self:ident, $config:ident, $cache:ident) => {{
        // skip the last token due to the input_end token
        let context_size = if $cache.output_tokens.is_empty() {
            $cache.input_tokens.len()
        } else {
            1
        };
        // tokens = input_tokens + output_tokens
        let tokens: Vec<u32> = $cache
            .input_tokens
            .iter()
            .chain($cache.output_tokens.iter())
            .cloned()
            .collect();

        let start_pos = tokens.len().saturating_sub(context_size);
        let context = &tokens[start_pos..];
        let input = Tensor::new(context, &$config.device)?.unsqueeze(0)?;
        let logits = $self.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

        let next_token = &$config.logits_processor.sample(&logits)?;
        let next_token = next_token.clone();
        $cache.output_tokens.push(next_token);

        if &next_token == &$config.eos_token {
            $cache.finished = true;
        }

        Ok(next_token)
    }};
}

macro_rules! decode_tokens {
    ($config:ident, $tokens:ident) => {
        if let Ok(text) = $config.tokenizer.decode(&$tokens, false) {
            Ok(text)
        } else {
            Err(E::msg("Error in decoding"))
        }
    };
}

impl TextGeneration for mistral::Model {
    fn get_next_token(
        &mut self,
        config: &mut GenerationConfig,
        cache: &mut GenerationCache,
    ) -> Result<u32> {
        get_next_token!(self, config, cache)
    }

    fn decode(&self, config: &mut GenerationConfig, tokens: &[u32]) -> Result<String> {
        decode_tokens!(config, tokens)
    }

    fn generate_tokens(&mut self, config: &mut GenerationConfig) -> Result<Vec<String>> {
        let tokens = config
            .tokenizer
            .encode(config.prompt.clone(), false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // sampling
        let mut cache = GenerationCache::new(Some(tokens));
        for _ in 0..config.max_new_tokens {
            self.get_next_token(config, &mut cache)?;

            if cache.finished {
                break;
            }
        }

        // clear kv cache
        self.clear_kv_cache();

        // decode the tokens
        let decoded = &cache
            .output_tokens
            .iter()
            .map(|&token| self.decode(config, &[token]))
            .collect::<Result<Vec<String>>>()?;

        Ok(decoded.clone())
    }
}

impl TextGeneration for mixtral::Model {
    fn get_next_token(
        &mut self,
        config: &mut GenerationConfig,
        cache: &mut GenerationCache,
    ) -> Result<u32> {
        get_next_token!(self, config, cache)
    }

    fn decode(&self, config: &mut GenerationConfig, tokens: &[u32]) -> Result<String> {
        decode_tokens!(config, tokens)
    }

    fn generate_tokens(&mut self, config: &mut GenerationConfig) -> Result<Vec<String>> {
        let tokens = config
            .tokenizer
            .encode(config.prompt.clone(), false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // sampling
        let mut cache = GenerationCache::new(Some(tokens));
        for _ in 0..config.max_new_tokens {
            self.get_next_token(config, &mut cache)?;

            if cache.finished {
                break;
            }
        }

        // clear kv cache
        self.clear_kv_cache();

        // decode the tokens
        let decoded = &cache
            .output_tokens
            .iter()
            .map(|&token| self.decode(config, &[token]))
            .collect::<Result<Vec<String>>>()?;

        Ok(decoded.clone())
    }
}

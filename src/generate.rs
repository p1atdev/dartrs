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

pub struct TextGeneration<T>
where
    T: CausalLM,
{
    model: T,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
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
    ) -> Self {
        let logits_processor = LogitsProcessor::from_sampling(
            seed,
            Sampling::TopKThenTopP {
                k: top_k.unwrap_or(100),
                p: top_p.unwrap_or(0.9),
                temperature: temp.unwrap_or(1.0),
            },
        );
        Self {
            model,
            tokenizer,
            logits_processor,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, max_new_tokens: usize) -> Result<()> {
        use std::io::Write;

        let mut tokens = self
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

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_vocab(true).get("<|eos|>").copied() {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|eos|> token"),
        };

        let start_gen = std::time::Instant::now();

        // sampling
        for index in 0..max_new_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.common_forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if let Ok(t) = self.tokenizer.decode(&[next_token], false) {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            if next_token == eos_token {
                break;
            } else {
                print!(", ");
                std::io::stdout().flush()?;
            }
        }

        let dt = start_gen.elapsed(); // finish

        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

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

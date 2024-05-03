use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use rand::random;

use candle_core::{DType, Device};

use hf_hub::{api::sync::Api, Repo, RepoType};

use dartrs::generation::{GenerationConfig, TextGeneration};
use dartrs::models::*;
use dartrs::prompt::compose_prompt_v2;
use dartrs::tags::{AspectRatioTag, IdentityTag, LengthTag, RatingTag};

#[derive(Debug, Clone, ValueEnum)]
enum ModelType {
    // #[clap(name = "llama")]
    // Llama,
    #[clap(name = "mistral")]
    Mistral,
    #[clap(name = "mixtral")]
    Mixtral,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(long, short, default_value = "mixtral")]
    model_type: ModelType,

    #[clap(long, default_value = "p1atdev/dart-v2-mixtral-160m-sft-2")]
    model_name: String,

    #[clap(long, default_value = "main")]
    revision: Option<String>,

    #[clap(long, default_value = "")]
    copyright: String,

    #[clap(long, default_value = "")]
    character: String,

    #[clap(long, value_enum, default_value = "long")]
    length: LengthTag,

    #[clap(long, value_enum, default_value = "tall")]
    aspect_ratio: AspectRatioTag,

    #[clap(long, value_enum, default_value = "sfw")]
    rating: RatingTag,

    #[clap(long, value_enum, default_value = "lax")]
    identity_level: IdentityTag,

    #[clap(long, short, default_value = "")]
    prompt: String,

    #[clap(long)]
    seed: Option<u64>,

    #[clap(long, default_value = "128")]
    max_new_tokens: usize,

    #[clap(long)]
    use_cuda: bool,
}

macro_rules! run {
    ($model:ident, $generation_config:ident) => {
        $model.run(&mut $generation_config)?; // generate text
    };
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let model_type = args.model_type;
    let model_name = args.model_name;
    let revision = args.revision;
    let max_new_tokens = args.max_new_tokens;

    let api = Api::new()?;

    let device = match args.use_cuda {
        true => Device::cuda_if_available(0),
        false => Ok(Device::Cpu),
    }?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    let start = std::time::Instant::now();

    let repo = ModelRepositoy::new(model_name.clone(), api.clone(), revision);

    let tokenizer = repo.load_tokenizer()?;

    let temperature = Some(1.0);
    let top_p = Some(0.9);
    let top_k = Some(100);
    let seed = args.seed.unwrap_or_else(|| random());

    // generate text
    let prompt = compose_prompt_v2(
        &args.copyright,
        &args.character,
        args.rating,
        args.aspect_ratio,
        args.length,
        args.identity_level,
        &args.prompt,
    );
    let mut generation_config = GenerationConfig::new(
        device.clone(),
        tokenizer,
        prompt,
        None,
        Some(max_new_tokens),
        temperature,
        top_p,
        top_k,
        Some(seed),
    );

    match model_type {
        ModelType::Mistral => {
            let mut model = MistralModelBuilder::load(&repo, dtype, &device)?;
            println!("loaded the model in {:?}", start.elapsed());

            run!(model, generation_config);
        }
        ModelType::Mixtral => {
            let mut model = MixtralModelBuilder::load(&repo, dtype, &device)?;
            println!("loaded the model in {:?}", start.elapsed());

            run!(model, generation_config);
        }
    }

    Ok(())
}

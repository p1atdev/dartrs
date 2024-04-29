use anyhow::{Error as E, Result};
use clap::Parser;
use rand::random;

use candle_core::{DType, Device};

use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

mod configs;
mod generate;
mod models;
mod prompt;
mod tags;

use configs::{DartV2Mistral, DartV2Mixtral};
use generate::TextGeneration;
use models::{MistralModelBuilder, MixtralModelBuilder, ModelBuilder};
use prompt::compose_prompt;
use tags::{AspectRatioTag, IdentityTag, LengthTag, RatingTag};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(long, short, default_value = "p1atdev/dart-v2-mistral-100m")]
    model_id: String,

    #[clap(long, default_value = "main")]
    revision: String,

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

    #[clap(long, value_enum, default_value = "none")]
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

fn main() -> Result<()> {
    let args = Args::parse();
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let model_id = args.model_id.to_string();
    let revision = args.revision.to_string();

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    // get model config.json
    println!("loading the config...");

    // get tokenizer.json
    let tokenizer_json = repo.get("tokenizer.json")?;
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_json).map_err(E::msg)?;

    let start = std::time::Instant::now();

    let device = match args.use_cuda {
        true => Device::cuda_if_available(0),
        false => Ok(Device::Cpu),
    }?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    // load model
    let model = MistralModelBuilder::dart_v2_100m(&api, dtype, &device).build()?;
    println!("loaded the model in {:?}", start.elapsed());

    // arguments
    let prompt = compose_prompt(
        &args.copyright,
        &args.character,
        args.rating,
        args.aspect_ratio,
        args.length,
        args.identity_level,
        &args.prompt,
    );
    let temperature = Some(1.0);
    let top_p = Some(0.9);
    let top_k = Some(100);
    let seed = args.seed.unwrap_or_else(|| random());

    // generate text
    let mut text_generation =
        TextGeneration::new(model, tokenizer, seed, temperature, top_p, top_k, &device);
    text_generation.run(&prompt, args.max_new_tokens)?;

    Ok(())
}

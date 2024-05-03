pub mod bindings;
pub mod configs;
pub mod generation;
pub mod models;
pub mod prompt;
pub mod tags;

use bindings::generation::*;
use bindings::models::*;
use bindings::prompt::*;
use bindings::tags::*;

use pyo3::prelude::*;

#[pymodule]
fn dartrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DartDType>()?;
    m.add_class::<DartDevice>()?;
    m.add_class::<DartV2Mistral>()?;
    m.add_class::<DartV2Mixtral>()?;
    m.add_class::<DartTokenizer>()?;
    m.add_class::<DartGenerationConfig>()?;
    m.add_class::<DartLengthTag>()?;
    m.add_class::<DartAspectRatioTag>()?;
    m.add_class::<DartRatingTag>()?;
    m.add_class::<DartIdentityTag>()?;
    m.add_class::<DartReservedTag>()?;
    m.add_function(wrap_pyfunction!(dart_compose_prompt_v2, m)?)?;

    Ok(())
}

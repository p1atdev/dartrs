pub mod bindings;
pub mod configs;
pub mod generation;
pub mod models;
pub mod polyfill;
pub mod prompt;
pub mod tags;

use bindings::generation::*;
use bindings::models::*;

use pyo3::prelude::*;

#[pymodule]
fn dartrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DartDType>()?;
    m.add_class::<DartDevice>()?;
    m.add_class::<DartV2Mistral>()?;
    m.add_class::<DartV2Mixtral>()?;
    m.add_class::<DartTokenizer>()?;
    m.add_class::<DartGenerationConfig>()?;
    Ok(())
}

use crate::generation::GenerationConfig;

use pyo3::prelude::*;

#[pyclass]
pub struct DartGenerationConfig(GenerationConfig);

use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub struct VectorIndex {
    dim: usize,
    data: Vec<f32>,
    count: usize,
}

#[napi]
impl VectorIndex {
    #[napi(constructor)]
    pub fn new(dim: u32) -> Result<Self> {
        if dim == 0 {
            return Err(Error::new(Status::InvalidArg, "dim must be more than 0"));
        }

        Ok(VectorIndex {
            dim: dim as usize,
            data: Vec::new(),
            count: 0,
        })
    }

    #[napi(getter)]
    pub fn size(&self) -> u32 {
        self.count as u32
    }

    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.dim as u32
    }

    #[napi(catch_unwind)]
    pub fn add_batch(&mut self, vectors: Float32Array) -> Result<u32> {
        if vectors.is_empty() {
            return Err(Error::new(Status::InvalidArg, "batch is empty"));
        }

        if vectors.len() % self.dim != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "length {} is not a multiple of dim {}",
                    vectors.len(),
                    self.dim
                ),
            ));
        }

        self.data.reserve(vectors.len());
        self.data.extend_from_slice(&vectors);
        self.count += vectors.len() / self.dim;
        Ok(self.count as u32)
    }
}

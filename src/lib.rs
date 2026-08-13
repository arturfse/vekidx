use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct Hit {
    pub index: u32,
    pub score: f64,
}

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

    /// Find the K most similar vectors.
    #[napi(catch_unwind)]
    pub fn search(&self, query: Float32Array, k: u32) -> Result<Vec<Hit>> {
        let q = self.check_query(&query)?;
        Ok(top_k(&self.data, self.dim, self.count, q, k as usize))
    }

    fn check_query<'a>(&self, query: &'a Float32Array) -> Result<&'a [f32]> {
        if query.len() != self.dim {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "query has {} values, while index needs {}",
                    query.len(),
                    self.dim
                ),
            ));
        }
        Ok(query)
    }
}

fn top_k(data: &[f32], dim: usize, count: usize, q: &[f32], k: usize) -> Vec<Hit> {
    let k = k.min(count);
    if k == 0 {
        return Vec::new();
    }

    let mut best: Vec<(f32, u32)> = Vec::with_capacity(count);
    for i in 0..count {
        best.push((dot(&data[i * dim..(i + 1) * dim], q), i as u32));
    }

    best.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
    best.truncate(k);
    best.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    best.into_iter()
        .map(|(s, i)| Hit {
            index: i,
            score: s as f64,
        })
        .collect()
}

/// Eight running totals let the CPU add eight pairs at once
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut acc = [0f32; 8];
    let chunks = a.len() / 8;

    for c in 0..chunks {
        let o = c * 8;
        for l in 0..8 {
            acc[l] += a[o + l] * b[o + l];
        }
    }

    let mut sum: f32 = acc.iter().sum();
    for i in chunks * 8..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_orders_by_score() {
        let data = vec![
            1.0, 0.0, // row 0
            0.0, 1.0, // row 1
            0.7, 0.7, // row 2
        ];
        let hits = top_k(&data, 2, 3, &[1.0, 0.0], 2);
        assert_eq!(hits[0].index, 0);
        assert_eq!(hits[1].index, 2);
    }

    #[test]
    fn top_k_with_k_zero_gives_nothing() {
        assert!(top_k(&[1.0, 0.0], 2, 1, &[1.0, 0.0], 0).is_empty());
    }

    #[test]
    fn top_k_on_an_empty_index_gives_nothing() {
        assert!(top_k(&[], 2, 0, &[1.0, 0.0], 5).is_empty());
    }

    #[test]
    fn top_k_asks_for_more_than_there_is() {
        assert_eq!(top_k(&[1.0, 0.0], 2, 1, &[1.0, 0.0], 99).len(), 1);
    }

    fn dot_slow(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn dot_matches_the_simple_version() {
        for len in [0usize, 1, 7, 8, 9, 384, 385] {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
            let b: Vec<f32> = (0..len).map(|i| 1.0 - i as f32 * 0.25).collect();
            let fast = dot(&a, &b);
            let slow = dot_slow(&a, &b);
            // Float adds round. Compare by ratio, not by a fixed difference.
            let scale = slow.abs().max(1.0);
            assert!(
                (fast - slow).abs() / scale < 1e-5,
                "len {len}: {fast} vs {slow}"
            );
        }
    }

    #[test]
    fn dot_handles_an_empty_slice() {
        assert_eq!(dot(&[], &[]), 0.0);
    }
}

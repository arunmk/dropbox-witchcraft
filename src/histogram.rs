pub struct Histogram {
    counts: Vec<u32>,
    max: u32,
    total: u32,
}

impl Histogram {
    pub fn new(max: u32) -> Self {
        Self {
            counts: vec![0; (max + 1) as usize],
            max,
            total: 0,
        }
    }

    /// Record a value (clamps if out of range)
    pub fn record(&mut self, v: u32) {
        let v = if v < self.max { v } else { self.max };
        self.counts[v as usize] += 1;
        self.total += 1;
    }

    /// Return quantile value (q ∈ [0.0, 1.0]), as u32.
    fn quantile(&self, q: f64) -> u32 {
        let rank: u32 = (q * self.total as f64).ceil() as u32;
        let mut cum: u32 = 0;
        for (val, &c) in self.counts.iter().enumerate() {
            cum += c;
            if cum >= rank {
                return val as u32;
            }
        }
        0
    }

    /// Convenience: p50
    pub fn p50(&self) -> u32 {
        self.quantile(0.5)
    }

    /// Convenience: p95
    pub fn p95(&self) -> u32 {
        self.quantile(0.95)
    }
}

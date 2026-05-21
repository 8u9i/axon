//! Sampling strategies for token generation.

use std::collections::BinaryHeap;

/// Sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for sampling. 0 = greedy (always pick highest probability).
    pub temperature: f32,
    /// Top-k: only sample from the top k tokens.
    pub top_k: usize,
    /// Top-p (nucleus): cumulative probability threshold.
    pub top_p: f32,
    /// Repetition penalty (>1.0 penalizes already-seen tokens).
    pub repeat_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.1,
        }
    }
}

/// Select an index based on a set of probabilities using greedy decoding.
pub fn greedy_sample(probs: &[f32]) -> usize {
    probs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Apply temperature to logits, then sample.
///
/// Returns the selected token index.
pub fn sample(logits: &[f32], params: &SamplingParams, rng: &mut impl FnMut() -> f32) -> usize {
    if logits.is_empty() {
        return 0;
    }

    // Apply temperature
    let mut scores: Vec<f32> = if params.temperature < 1e-6 {
        // Greedy
        return greedy_sample(logits);
    } else {
        logits.iter().map(|&l| l / params.temperature).collect()
    };

    // Apply softmax
    let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for v in scores.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum <= 0.0 {
        return 0;
    }
    let inv_sum = 1.0 / sum;
    for v in scores.iter_mut() {
        *v *= inv_sum;
    }

    // Top-k filtering
    let k = params.top_k.min(scores.len());
    if k < scores.len() {
        let mut heap: BinaryHeap<OrderedFloatIdx> =
            scores.iter().enumerate().map(|(i, &v)| OrderedFloatIdx(v, i)).collect();
        let mut threshold = 0.0f32;
        for _ in 0..k {
            if let Some(item) = heap.pop() {
                threshold = item.0;
            }
        }
        // Zero out scores below threshold
        for v in scores.iter_mut() {
            if *v < threshold {
                *v = 0.0;
            }
        }
        // Re-normalize
        let s: f32 = scores.iter().sum();
        if s > 0.0 {
            let inv = 1.0 / s;
            for v in scores.iter_mut() {
                *v *= inv;
            }
        }
    }

    // Top-p (nucleus) filtering
    if params.top_p < 1.0 {
        let mut sorted: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        sorted.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumsum = 0.0f32;
        let mut cutoff = 0.0f32;
        for &(_, v) in &sorted {
            cumsum += v;
            if cumsum >= params.top_p {
                cutoff = v;
                break;
            }
        }
        for v in scores.iter_mut() {
            if *v < cutoff {
                *v = 0.0;
            }
        }
        let s: f32 = scores.iter().sum();
        if s > 0.0 {
            let inv = 1.0 / s;
            for v in scores.iter_mut() {
                *v *= inv;
            }
        }
    }

    // Sample from the filtered distribution
    let r = rng();
    let mut cumsum = 0.0f32;
    for (i, &v) in scores.iter().enumerate() {
        cumsum += v;
        if r <= cumsum {
            return i;
        }
    }
    // Fallback
    scores.len() - 1
}

/// Wrapper for (f32, usize) to implement Ord for BinaryHeap (max-heap by f32).
#[derive(PartialEq)]
struct OrderedFloatIdx(f32, usize);

impl Eq for OrderedFloatIdx {}

impl PartialOrd for OrderedFloatIdx {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloatIdx {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

use tract_onnx::prelude::*;
use tract_ndarray::{Array2, Array3};
use atb::rl::obs::OBS_TOTAL;
use atb::rl::action::ACTION_WAIT;

type Model = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

struct LstmState {
    h: Vec<f32>,
    c: Vec<f32>,
    n_layers:    usize,
    hidden_size: usize,
}

impl LstmState {
    fn zero(n_layers: usize, hidden_size: usize) -> Self {
        let len = n_layers * hidden_size;
        Self { h: vec![0.0; len], c: vec![0.0; len], n_layers, hidden_size }
    }

    fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }
}

pub struct OnnxPolicy {
    model: Model,
    lstm:  Option<LstmState>,
}

impl OnnxPolicy {
    pub fn load(path: &str) -> TractResult<Self> {
        // Check for LSTM sidecar metadata written by export.py.
        let meta_path = std::path::Path::new(path)
            .with_file_name("policy_meta.json");

        let lstm_dims: Option<(usize, usize)> = meta_path.exists()
            .then(|| {
                let text = std::fs::read_to_string(&meta_path).ok()?;
                let v: serde_json::Value = serde_json::from_str(&text).ok()?;
                let n = v["lstm_n_layers"].as_u64()? as usize;
                let h = v["lstm_hidden_size"].as_u64()? as usize;
                Some((n, h))
            })
            .flatten();

        let mut builder = tract_onnx::onnx().model_for_path(path)?;
        builder = builder.with_input_fact(0, f32::fact([1, OBS_TOTAL]).into())?;

        let lstm = if let Some((n_layers, hidden_size)) = lstm_dims {
            builder = builder
                .with_input_fact(1, f32::fact([n_layers, 1, hidden_size]).into())?
                .with_input_fact(2, f32::fact([n_layers, 1, hidden_size]).into())?;
            Some(LstmState::zero(n_layers, hidden_size))
        } else {
            None
        };

        let model = builder.into_optimized()?.into_runnable()?;
        Ok(Self { model, lstm })
    }

    pub fn reset(&mut self) {
        if let Some(ref mut s) = self.lstm { s.reset(); }
    }

    /// Run the policy network and return the raw action logits (length ACTION_SIZE),
    /// or None on any failure. Updates LSTM state if the model has one.
    fn logits(&mut self, obs: &[f32]) -> Option<Vec<f32>> {
        debug_assert_eq!(obs.len(), OBS_TOTAL,
            "obs length mismatch: got {}, expected {OBS_TOTAL}", obs.len());

        let obs_arr = Array2::from_shape_vec([1, OBS_TOTAL], obs.to_vec()).ok()?;

        let outputs = if let Some(ref s) = self.lstm {
            let shape = [s.n_layers, 1, s.hidden_size];
            let h_arr = Array3::from_shape_vec(shape, s.h.clone()).ok()?;
            let c_arr = Array3::from_shape_vec(shape, s.c.clone()).ok()?;
            self.model.run(tvec!(
                Tensor::from(obs_arr).into(),
                Tensor::from(h_arr).into(),
                Tensor::from(c_arr).into(),
            ))
        } else {
            self.model.run(tvec!(Tensor::from(obs_arr).into()))
        };

        let outputs = outputs.ok()?;

        // Update LSTM state from h_out (index 1) and c_out (index 2).
        if let Some(ref mut s) = self.lstm {
            if outputs.len() >= 3 {
                if let Ok(h_view) = outputs[1].to_array_view::<f32>() {
                    s.h = h_view.iter().copied().collect();
                }
                if let Ok(c_view) = outputs[2].to_array_view::<f32>() {
                    s.c = c_view.iter().copied().collect();
                }
            }
        }

        outputs[0].as_slice::<f32>().ok().map(|s| s.to_vec())
    }

    /// Greedy action over all logits (no masking). Kept as a fallback API; the
    /// viewer uses `act_masked` to match the policy's MaskablePPO training.
    #[allow(dead_code)]
    pub fn act(&mut self, obs: &[f32]) -> u32 {
        let logits = match self.logits(obs) { Some(l) => l, None => return ACTION_WAIT };
        argmax(logits.iter().copied().enumerate())
    }

    /// Greedy action restricted to valid actions (`mask[i] == true`). Mirrors the
    /// MaskablePPO inference the policy was trained under — without it the argmax
    /// could pick a masked, provably-useless action. Falls back to Wait if no
    /// action is valid.
    pub fn act_masked(&mut self, obs: &[f32], mask: &[bool]) -> u32 {
        let logits = match self.logits(obs) { Some(l) => l, None => return ACTION_WAIT };
        argmax(
            logits.iter().copied().enumerate()
                .filter(|(i, _)| mask.get(*i).copied().unwrap_or(false)),
        )
    }
}

/// Index of the max-logit element, or ACTION_WAIT if the iterator is empty.
fn argmax(it: impl Iterator<Item = (usize, f32)>) -> u32 {
    it.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(ACTION_WAIT)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard: tract must be able to analyse + optimise the exported
    /// graph. An open-ended cluster slice in the extractor used to leave the
    /// cluster_head Gemm's input dim symbolic, failing `into_optimized()` and
    /// forcing the viewer onto the BT-agent fallback.
    #[test]
    fn loads_exported_policy() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/model/policy.onnx");
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: no exported policy at {path}");
            return;
        }
        OnnxPolicy::load(path).expect("tract should analyse + load the exported policy");
    }
}

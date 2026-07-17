use tract_onnx::prelude::*;
use tract_ndarray::Array2;
use atb::rl::obs::OBS_TOTAL;
use atb::rl::action::ACTION_WAIT;

type Model = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct OnnxPolicy {
    model: Model,
}

impl OnnxPolicy {
    pub fn load(path: &str) -> TractResult<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact([1, OBS_TOTAL]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    /// Run the policy network and return the raw action logits (length ACTION_SIZE),
    /// or None on any failure. The policy is feedforward — no recurrent state.
    fn logits(&self, obs: &[f32]) -> Option<Vec<f32>> {
        debug_assert_eq!(obs.len(), OBS_TOTAL,
            "obs length mismatch: got {}, expected {OBS_TOTAL}", obs.len());

        let obs_arr = Array2::from_shape_vec([1, OBS_TOTAL], obs.to_vec()).ok()?;
        let outputs = self.model.run(tvec!(Tensor::from(obs_arr).into())).ok()?;
        outputs[0].as_slice::<f32>().ok().map(|s| s.to_vec())
    }

    /// Greedy action restricted to valid actions (`mask[i] == true`). Mirrors the
    /// MaskablePPO inference the policy was trained under — without it the argmax
    /// could pick a masked, provably-useless action. Falls back to Wait if no
    /// action is valid.
    pub fn act_masked(&self, obs: &[f32], mask: &[bool]) -> u32 {
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

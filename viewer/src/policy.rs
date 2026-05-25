use tract_onnx::prelude::*;
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

    pub fn act(&self, obs: &[f32]) -> u32 {
        debug_assert_eq!(
            obs.len(), OBS_TOTAL,
            "obs length mismatch: got {}, expected {OBS_TOTAL}", obs.len()
        );

        let input: Tensor = match tract_ndarray::Array2::from_shape_vec(
            [1, OBS_TOTAL],
            obs.to_vec(),
        ) {
            Ok(a)  => a.into(),
            Err(_) => return ACTION_WAIT,
        };

        let outputs = match self.model.run(tvec!(input.into())) {
            Ok(o)  => o,
            Err(_) => return ACTION_WAIT,
        };

        let logits: &[f32] = match outputs[0].as_slice() {
            Ok(s)  => s,
            Err(_) => return ACTION_WAIT,
        };

        logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(ACTION_WAIT)
    }
}
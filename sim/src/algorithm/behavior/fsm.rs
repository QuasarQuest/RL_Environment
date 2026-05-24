// src/algorithm/behavior/fsm.rs
//
// Generic deterministic FSM primitive.
// Zero knowledge of agents, planners, or game logic.

#![allow(dead_code)]

// ── Fsm ───────────────────────────────────────────────────────────────────────

pub struct Fsm<S: FsmState> {
    state: S,
}

impl<S: FsmState> Fsm<S> {
    pub fn new(initial: S) -> Self {
        Self { state: initial }
    }

    pub fn state(&self) -> &S {
        &self.state
    }

    /// Mutable access — use only to mutate data *within* the current variant,
    /// not to change variant. Use `transition` for variant changes.
    pub fn state_mut(&mut self) -> &mut S {
        &mut self.state
    }

    pub fn transition(&mut self, next: S) {
        let from = self.state.name();
        let to   = next.name();
        if from != to {
            let _ = (from, to); // transition logged externally if needed
        }
        self.state = next;
    }

    /// Take the current state by value, transform it, write it back.
    /// Used when a state variant needs to be updated in place.
    pub fn replace<F: FnOnce(S) -> S>(&mut self, f: F) {
        let old = std::mem::replace(&mut self.state, S::idle());
        self.state = f(old);
    }

    pub fn name(&self) -> &'static str {
        self.state.name()
    }
}

// ── FsmState trait ────────────────────────────────────────────────────────────

pub trait FsmState: Sized + Send + Sync {
    fn name(&self) -> &'static str;
    /// Construct the idle/reset variant. Used by `Fsm::replace` to vacate
    /// the state slot temporarily so the old value can be moved.
    fn idle() -> Self;
}

// ── Guard helper ──────────────────────────────────────────────────────────────

pub struct Guard<I, F: Fn(&I) -> bool>(F, std::marker::PhantomData<fn(&I)>);

impl<I, F: Fn(&I) -> bool> Guard<I, F> {
    pub fn new(f: F) -> Self { Self(f, std::marker::PhantomData) }
    pub fn check(&self, input: &I) -> bool { (self.0)(input) }
}
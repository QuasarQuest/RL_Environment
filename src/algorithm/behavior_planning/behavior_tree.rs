// src/algorithm/behavior_planning/behavior_tree.rs
//
// Generic synchronous behaviour tree kernel.
//
// This file knows nothing about agents, actions, observations, or planners.
// It is parameterised over two types:
//
//   I  — the read-only input passed into every tick (e.g. &Observation)
//   O  — the output a leaf may produce (e.g. Action)
//
// Consumers (e.g. AStarAgent) instantiate BtNode<Observation<'_>, Action>
// and build trees with the builder helpers at the bottom.
//
// Execution model
// ───────────────
//   Failure  — this node could not contribute; try the next sibling (Selector)
//              or abort the parent (Sequence).
//   Success  — this node finished cleanly; continue to next sibling (Sequence)
//              or short-circuit the parent (Selector).
//   Running  — this node produced an output this tick; stop the entire tree.
//              The caller reads `out` to get the action.
//
// No heap allocation occurs in compositors. Leaf closures may allocate if
// they choose to — that is their contract with the caller, not ours.

// ── Status ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    /// Node completed without producing output. Propagates upward.
    Success,
    /// Node could not act. Propagates upward.
    Failure,
    /// Node produced output this tick. Tree evaluation stops immediately.
    Running,
}

// ── Core trait ────────────────────────────────────────────────────────────────

/// A single node in the behaviour tree.
///
/// `I` — immutable context read by every node (observation, world state, …).
/// `O` — the output type a leaf writes when it acts (action, command, …).
///
/// Implementors must be `Send + Sync` so the tree can live inside a Bevy
/// `Component` (which requires `Send + Sync` on its contents).
pub trait BtNode<I, O>: Send + Sync {
    fn tick(&mut self, input: &I, out: &mut Option<O>) -> Status;
}

// ── Compositors ───────────────────────────────────────────────────────────────

/// Tries children left-to-right.
/// Returns the first non-Failure status (Success or Running).
/// Returns Failure only when every child fails.
pub struct Selector<I, O>(pub Vec<Box<dyn BtNode<I, O>>>);

impl<I, O> BtNode<I, O> for Selector<I, O>
where
    I: Send + Sync,
    O: Send + Sync,
{
    fn tick(&mut self, input: &I, out: &mut Option<O>) -> Status {
        for child in &mut self.0 {
            match child.tick(input, out) {
                Status::Failure => continue,
                other           => return other,
            }
        }
        Status::Failure
    }
}

/// Runs children left-to-right.
/// Returns Failure immediately when any child fails.
/// Returns Success only when every child succeeds.
/// Running from any child stops the entire tree.
pub struct Sequence<I, O>(pub Vec<Box<dyn BtNode<I, O>>>);

impl<I, O> BtNode<I, O> for Sequence<I, O>
where
    I: Send + Sync,
    O: Send + Sync,
{
    fn tick(&mut self, input: &I, out: &mut Option<O>) -> Status {
        for child in &mut self.0 {
            match child.tick(input, out) {
                Status::Success => continue,
                other           => return other,
            }
        }
        Status::Success
    }
}

/// Inverts the status of its single child.
/// Running passes through unchanged (the output is already written).
pub struct Inverter<I, O>(pub Box<dyn BtNode<I, O>>);

impl<I, O> BtNode<I, O> for Inverter<I, O>
where
    I: Send + Sync,
    O: Send + Sync,
{
    fn tick(&mut self, input: &I, out: &mut Option<O>) -> Status {
        match self.0.tick(input, out) {
            Status::Success => Status::Failure,
            Status::Failure => Status::Success,
            Status::Running => Status::Running,
        }
    }
}

// ── Leaves ────────────────────────────────────────────────────────────────────

/// Stateless guard / condition node.
/// Calls `f(input)` and maps `true → Success`, `false → Failure`.
/// Never writes to `out`, never returns Running.
pub struct Condition<I, F>(pub F, std::marker::PhantomData<fn(&I)>);

impl<I, F> Condition<I, F> {
    pub fn new(f: F) -> Self { Self(f, std::marker::PhantomData) }
}

impl<I, O, F> BtNode<I, O> for Condition<I, F>
where
    I:  Send + Sync,
    O:  Send + Sync,
    F:  Fn(&I) -> bool + Send + Sync,
{
    fn tick(&mut self, input: &I, _out: &mut Option<O>) -> Status {
        if (self.0)(input) { Status::Success } else { Status::Failure }
    }
}

/// Stateful action leaf.
/// The closure `f` may write to `out` and returns the resulting Status.
/// Return Running + write out → acted this tick.
/// Return Failure            → could not act; try next sibling.
pub struct Leaf<I, O, F>(pub F, std::marker::PhantomData<fn(&I) -> O>);

impl<I, O, F> Leaf<I, O, F> {
    pub fn new(f: F) -> Self { Self(f, std::marker::PhantomData) }
}

impl<I, O, F> BtNode<I, O> for Leaf<I, O, F>
where
    I:  Send + Sync,
    O:  Send + Sync,
    F:  FnMut(&I, &mut Option<O>) -> Status + Send + Sync,
{
    fn tick(&mut self, input: &I, out: &mut Option<O>) -> Status {
        (self.0)(input, out)
    }
}

// ── Builder helpers ───────────────────────────────────────────────────────────
//
// These free functions let agent code write trees without spelling out
// the full generic types every time.

pub fn selector<I, O>(children: Vec<Box<dyn BtNode<I, O>>>) -> Box<dyn BtNode<I, O>>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    Box::new(Selector(children))
}

pub fn sequence<I, O>(children: Vec<Box<dyn BtNode<I, O>>>) -> Box<dyn BtNode<I, O>>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    Box::new(Sequence(children))
}

pub fn invert<I, O>(child: Box<dyn BtNode<I, O>>) -> Box<dyn BtNode<I, O>>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    Box::new(Inverter(child))
}

pub fn cond<I, O>(
    f: impl Fn(&I) -> bool + Send + Sync + 'static,
) -> Box<dyn BtNode<I, O>>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    Box::new(Condition::new(f))
}

pub fn leaf<I, O>(
    f: impl FnMut(&I, &mut Option<O>) -> Status + Send + Sync + 'static,
) -> Box<dyn BtNode<I, O>>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    Box::new(Leaf::new(f))
}
// src/algorithm/behavior/behavior_tree.rs
//
// Generic synchronous behaviour tree kernel.
// Parameterised over I (input) and O (output).
// Consumers instantiate BtNode<Observation, Action> and build trees
// with the builder helpers at the bottom.

#![allow(dead_code)]

// ── Status ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Success,
    Failure,
    Running,
}

// ── Core trait ────────────────────────────────────────────────────────────────

pub trait BtNode<I, O>: Send + Sync {
    fn tick(&mut self, input: &I, out: &mut Option<O>) -> Status;
}

// ── Compositors ───────────────────────────────────────────────────────────────

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

pub struct Condition<I, F>(pub F, std::marker::PhantomData<fn(&I)>);

impl<I, F> Condition<I, F> {
    pub fn new(f: F) -> Self { Self(f, std::marker::PhantomData) }
}

impl<I, O, F> BtNode<I, O> for Condition<I, F>
where
    I: Send + Sync,
    O: Send + Sync,
    F: Fn(&I) -> bool + Send + Sync,
{
    fn tick(&mut self, input: &I, _out: &mut Option<O>) -> Status {
        if (self.0)(input) { Status::Success } else { Status::Failure }
    }
}

pub struct Leaf<I, O, F>(pub F, std::marker::PhantomData<fn(&I) -> O>);

impl<I, O, F> Leaf<I, O, F> {
    pub fn new(f: F) -> Self { Self(f, std::marker::PhantomData) }
}

impl<I, O, F> BtNode<I, O> for Leaf<I, O, F>
where
    I: Send + Sync,
    O: Send + Sync,
    F: FnMut(&I, &mut Option<O>) -> Status + Send + Sync,
{
    fn tick(&mut self, input: &I, out: &mut Option<O>) -> Status {
        (self.0)(input, out)
    }
}

// ── Builder helpers ───────────────────────────────────────────────────────────

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
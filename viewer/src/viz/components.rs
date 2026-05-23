use bevy::prelude::*;

#[derive(Component, Clone, Debug)]
pub struct AgentLabel(pub String);

impl AgentLabel {
    pub fn new(name: impl Into<String>) -> Self { Self(name.into()) }
}

#[derive(Component, Clone, Debug)]
pub struct AgentInfo {
    pub strategy: &'static str,
}

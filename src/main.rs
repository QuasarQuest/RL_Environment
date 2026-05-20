// src/main.rs
//
// Thin binary entry point. All module declarations and the App builder
// live in the library crate (`atb`); this file only exists so `cargo run`
// has something to launch. Adding modules here would create a parallel
// module tree and compile every file twice.

fn main() {
    atb::run();
}

// TODO: Save Stats after ML run
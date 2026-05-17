// src/rl/action.rs
//
// Maps discrete integer actions (0..25) to the sim's Action enum.
// This is the action space the neural net outputs.
//
// Layout:
//   0..7   Move       (N, S, E, W, NE, NW, SE, SW)
//   8..15  Attack     (N, S, E, W, NE, NW, SE, SW)
//   16..23 RangedAttack (N, S, E, W, NE, NW, SE, SW)
//   24     Drop
//   25     Wait

use crate::agent::action::{Action, Dir};

pub const ACTION_SIZE: usize = 26;

/// Convert a neural net output integer to a sim Action.
/// Panics on out-of-range — Python side must clamp to 0..ACTION_SIZE.
pub fn int_to_action(action: u32) -> Action {
    match action {
        0  => Action::Move(Dir::N),
        1  => Action::Move(Dir::S),
        2  => Action::Move(Dir::E),
        3  => Action::Move(Dir::W),
        4  => Action::Move(Dir::NE),
        5  => Action::Move(Dir::NW),
        6  => Action::Move(Dir::SE),
        7  => Action::Move(Dir::SW),

        8  => Action::Attack(Dir::N),
        9  => Action::Attack(Dir::S),
        10 => Action::Attack(Dir::E),
        11 => Action::Attack(Dir::W),
        12 => Action::Attack(Dir::NE),
        13 => Action::Attack(Dir::NW),
        14 => Action::Attack(Dir::SE),
        15 => Action::Attack(Dir::SW),

        16 => Action::RangedAttack(Dir::N),
        17 => Action::RangedAttack(Dir::S),
        18 => Action::RangedAttack(Dir::E),
        19 => Action::RangedAttack(Dir::W),
        20 => Action::RangedAttack(Dir::NE),
        21 => Action::RangedAttack(Dir::NW),
        22 => Action::RangedAttack(Dir::SE),
        23 => Action::RangedAttack(Dir::SW),

        24 => Action::Drop,
        25 => Action::Wait,

        _ => panic!("Invalid RL action index: {action} (must be 0..{ACTION_SIZE})"),
    }
}

/// Convert a sim Action back to an integer — useful for logging/debugging.
pub fn action_to_int(action: Action) -> u32 {
    match action {
        Action::Move(Dir::N)  => 0,
        Action::Move(Dir::S)  => 1,
        Action::Move(Dir::E)  => 2,
        Action::Move(Dir::W)  => 3,
        Action::Move(Dir::NE) => 4,
        Action::Move(Dir::NW) => 5,
        Action::Move(Dir::SE) => 6,
        Action::Move(Dir::SW) => 7,

        Action::Attack(Dir::N)  => 8,
        Action::Attack(Dir::S)  => 9,
        Action::Attack(Dir::E)  => 10,
        Action::Attack(Dir::W)  => 11,
        Action::Attack(Dir::NE) => 12,
        Action::Attack(Dir::NW) => 13,
        Action::Attack(Dir::SE) => 14,
        Action::Attack(Dir::SW) => 15,

        Action::RangedAttack(Dir::N)  => 16,
        Action::RangedAttack(Dir::S)  => 17,
        Action::RangedAttack(Dir::E)  => 18,
        Action::RangedAttack(Dir::W)  => 19,
        Action::RangedAttack(Dir::NE) => 20,
        Action::RangedAttack(Dir::NW) => 21,
        Action::RangedAttack(Dir::SE) => 22,
        Action::RangedAttack(Dir::SW) => 23,

        Action::Drop => 24,
        Action::Wait => 25,
    }
}
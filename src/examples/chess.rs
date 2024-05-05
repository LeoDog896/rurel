/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::path::PathBuf;

#[cfg(feature = "dqn")]
use rurel::dqn::DQNAgentTrainer;
use rurel::{mdp::{Agent, State}, strategy::terminate::TerminationStrategy};
use shakmaty::{Chess, Color, EnPassantMode, Move, Position, Role, Square};
use clap::Parser;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct ChessState(Chess);

fn u32_to_square(n: u32) -> Square {
    Square::ALL[n as usize]
}

fn u32_to_role(n: u32) -> Role {
    match n {
        0 => Role::Pawn,
        1 => Role::Knight,
        2 => Role::Bishop,
        3 => Role::Rook,
        4 => Role::Queen,
        5 => Role::King,
        _ => panic!("Invalid role {}", n),
    }
}

/// Split a u64 into two u32s.
fn split_u64(n: u64) -> (u32, u32) {
    ((n & 0xFFFF_FFFF) as u32, (n >> 32) as u32)
}

macro_rules! generate_from_chess_state {
    (
        $val:ident, 
        $array:ident,
        $(($field:ident, $index:expr)),*
    ) => {
        {
            $(
                let ($field, _b) = split_u64($val.$field().0);
                $array[2 * $index] = $field as f32;
                $array[2 * $index + 1] = _b as f32;
            )*
        }
    };
}

impl From<ChessState> for [f32; 21] {
    fn from(val: ChessState) -> Self {
        let mut array = [0.0; 21];
        let board = val.0.board();
        // fill the first 16 elements with the bitboards
        generate_from_chess_state!(
            board, array,
            (pawns, 0),
            (knights, 1),
            (bishops, 2),
            (rooks, 3),
            (queens, 4),
            (kings, 5),
            (white, 6),
            (black, 7)
        );
        // then the turn
        array[16] = if val.0.turn() == Color::White { 0.0 } else { 1.0 };
        // then halfmove clock
        array[17] = val.0.halfmoves() as f32;
        // then en passant square
        array[18] = val.0.ep_square(EnPassantMode::Legal).map(|x| x as u32 as f32 + 1.0).unwrap_or(0.0);
        // then castling rights
        let (castling_a, castling_b) = split_u64(val.0.into_setup(EnPassantMode::Legal).castling_rights.0);
        array[19] = castling_a as f32;
        array[20] = castling_b as f32;

        array
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct ChessAction(Move);

// Into float array has to be implemented for the action,
// so that the DQN can use it.
impl From<ChessAction> for [f32; 6] {
    fn from(val: ChessAction) -> Self {
        match val.0 {
            Move::Normal {
                role,
                from,
                capture,
                to,
                promotion,
            } => {
                [
                    0.0, // Normal
                    role as u32 as f32,
                    from as u32 as f32,
                    match capture {
                        Some(role) => role as u32 as f32 + 1.0,
                        None => 0.0,
                    },
                    to as u32 as f32,
                    match promotion {
                        Some(role) => role as u32 as f32 + 1.0,
                        None => 0.0,
                    },
                ]
            }
            Move::EnPassant { from, to } => {
                [
                    1.0, // En Passant
                    0.0,
                    from as u32 as f32,
                    0.0,
                    to as u32 as f32,
                    0.0,
                ]
            }
            Move::Castle { king, rook } => {
                [
                    2.0, // Castle
                    0.0,
                    king as u32 as f32,
                    0.0,
                    rook as u32 as f32,
                    0.0,
                ]
            }
            Move::Put { role, to } => {
                [
                    3.0, // Put
                    role as u32 as f32,
                    0.0,
                    0.0,
                    to as u32 as f32,
                    0.0,
                ]
            }
        }
    }
}

// From float array has to be implemented for the action,
// because output of the DQN is a float array like [0.1, 0.2, 0.1, 0.1]
impl From<[f32; 6]> for ChessAction {
    fn from(v: [f32; 6]) -> Self {
        println!("{:?}", v);
        match v[0] as u32 {
            0 => {
                let role = v[1] as u32;
                let from = v[2] as u32;
                let capture = if v[3] as u32 == 0 {
                    None
                } else {
                    Some(v[3] as u32 - 1)
                };
                let to = v[4] as u32;
                let promotion = if v[5] as u32 == 0 {
                    None
                } else {
                    Some(v[5] as u32 - 1)
                };
                ChessAction(Move::Normal {
                    role: u32_to_role(role),
                    from: u32_to_square(from),
                    capture: capture.map(|x| u32_to_role(x as u32 - 1)),
                    to: to.try_into().unwrap(),
                    promotion: promotion.map(|x| u32_to_role(x as u32 - 1)),
                })
            }
            1 => {
                let from = v[2] as u32;
                let to = v[4] as u32;
                ChessAction(Move::EnPassant {
                    from: u32_to_square(from),
                    to: u32_to_square(to),
                })
            }
            2 => {
                let king = v[2] as u32;
                let rook = v[4] as u32;
                ChessAction(Move::Castle {
                    king: u32_to_square(king),
                    rook: u32_to_square(rook),
                })
            }
            3 => {
                let role = v[1] as u32;
                let to = v[4] as u32;
                ChessAction(Move::Put {
                    role: u32_to_role(role),
                    to: u32_to_square(to),
                })
            }
            _ => panic!("Invalid action"),
        }
    }
}

impl State for ChessState {
    type A = ChessAction;

    fn reward(&self) -> f64 {
        match self.0.outcome() {
            Some(outcome) => match outcome {
                shakmaty::Outcome::Decisive { winner, .. } => {
                    if winner == self.0.turn() {
                        -20.0
                    } else {
                        40.0
                    }
                }
                shakmaty::Outcome::Draw { .. } => -10.0,
            },
            None => -10.0,
        }
    }

    fn actions(&self) -> Vec<ChessAction> {
        self.0.legal_moves().iter().cloned().map(ChessAction).collect()
    }
}

struct ChessAgent(ChessState);

impl Agent<ChessState> for ChessAgent {
    fn current_state(&self) -> &ChessState {
        &self.0
    }

    fn take_action(&mut self, action: &ChessAction) {
        self.0 = ChessState(self.0.0.clone().play(&action.0).unwrap());
    }
}

struct ChessTermination;

impl TerminationStrategy<ChessState> for ChessTermination {
    fn should_stop(&mut self, state: &ChessState) -> bool {
        state.0.outcome().is_some() || state.0.halfmoves() >= 100
    }
}

#[derive(Parser)]
struct Cli {
    /// The path to the file to save the model to.
    file: PathBuf,

    /// The number of trials to run.
    #[arg(short, long, default_value = "10000")]
    trials: i32,
}

#[cfg(feature = "dqn")]
fn main() {
    use indicatif::ProgressIterator;
    use rurel::strategy::explore::RandomExploration;

    let cli = Cli::parse();

    // check if file exists; if so, load the model
    let trainer = if cli.file.exists() {
        let mut trainer = DQNAgentTrainer::<ChessState, 21, 6, 64>::new(0.9, 1e-3);
        trainer.load(&cli.file.to_str().unwrap()).unwrap();
        trainer
    } else {
        let initial_state = ChessState(Chess::default());

        let mut trainer = DQNAgentTrainer::<ChessState, 21, 6, 64>::new(0.9, 1e-3);
        for _ in (0..cli.trials).progress() {
            let mut agent = ChessAgent(initial_state.clone());
            trainer.train(
                &mut agent,
                &mut ChessTermination,
                &RandomExploration,
            );
        }

        trainer.save(&cli.file.to_str().unwrap()).unwrap();

        trainer
    };

    // Play against the agent
    let mut state = ChessState(Chess::default());

    loop {
        println!("{:?}", state.0.board());
        let legal_moves = state.0.legal_moves();
        if legal_moves.is_empty() {
            println!("Game over");
            break;
        }

        let action = if state.0.turn() == Color::White {
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
            let action = legal_moves
                .iter()
                .find(|m| m.to_string() == input.trim());

            let action = match action {
                Some(action) => action,
                None => {
                    println!("Invalid move; valid moves are:");
                    for m in legal_moves.iter() {
                        println!("{}", m);
                    }
                    continue;
                }
            };

            ChessAction(action.clone())
        } else {
            let action = trainer
                .best_action(&state)
                .expect("No legal moves available");
            
            action
        };

        println!(
            "{} played: {}",
            state.0.turn(),
            action.0
        );
        state = ChessState(state.0.clone().play(&action.0).unwrap());
    }

}

#[cfg(not(feature = "dqn"))]
fn main() {
    panic!("Use the 'dqn' feature to run this example");
}

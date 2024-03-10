/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::num::NonZeroU32;

#[cfg(feature = "dqn")]
use rurel::dqn::DQNAgentTrainer;
use rurel::{mdp::{Agent, State}, strategy::terminate::TerminationStrategy};
use shakmaty::{Bitboard, Board, ByColor, ByRole, CastlingMode, Chess, Color, EnPassantMode, FromSetup, Move, Position, Setup};


#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct ChessState(Chess);

/// Split a u64 into two u32s.
/// Inverse of [join_to_u64].
fn split_u64(n: u64) -> (u32, u32) {
    ((n & 0xFFFF_FFFF) as u32, (n >> 32) as u32)
}

/// Join two u32s into a u64.
/// Inverse of [split_u64].
fn join_to_u64(a: u32, b: u32) -> u64 {
    (a as u64) | ((b as u64) << 32)
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

impl From<ChessState> for [f32; 24] {
    fn from(val: ChessState) -> Self {
        let mut array = [0.0; 24];
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
        // then fullmove number
        array[18] = u32::from(val.0.fullmoves()) as f32;
        // then en passant square
        array[19] = val.0.ep_square(EnPassantMode::Legal).map(|x| x as u32 as f32).unwrap_or(0.0);
        // then promoted bitboard
        let (promoted_a, promoted_b) = split_u64(val.0.promoted().0);
        array[20] = promoted_a as f32;
        array[21] = promoted_b as f32;
        // then castling rights
        let (castling_a, castling_b) = split_u64(val.0.into_setup(EnPassantMode::Legal).castling_rights.0);
        array[22] = castling_a as f32;
        array[23] = castling_b as f32;

        array
    }
}

// From float array has to be implemented for the DQN state
impl From<[f32; 24]> for ChessState {
    fn from(v: [f32; 24]) -> Self {
        ChessState(Chess::from_setup(Setup {
            board: Board::from_bitboards(
                ByRole {
                    pawn: Bitboard(join_to_u64(v[0] as u32, v[1] as u32)),
                    knight: Bitboard(join_to_u64(v[2] as u32, v[3] as u32)),
                    bishop: Bitboard(join_to_u64(v[4] as u32, v[5] as u32)),
                    rook: Bitboard(join_to_u64(v[6] as u32, v[7] as u32)),
                    queen: Bitboard(join_to_u64(v[8] as u32, v[9] as u32)),
                    king: Bitboard(join_to_u64(v[10] as u32, v[11] as u32)),
                },
                ByColor {
                    white: Bitboard(join_to_u64(v[12] as u32, v[13] as u32)),
                    black: Bitboard(join_to_u64(v[14] as u32, v[15] as u32)),
                },
            ),
            turn: if v[16] == 0.0 { Color::White } else { Color::Black },
            halfmoves: v[17] as u32,
            fullmoves: NonZeroU32::new(v[18] as u32).unwrap(),
            ep_square: if v[19] == 0.0 { None } else { Some((v[19] as u32).try_into().unwrap()) },
            promoted: Bitboard(join_to_u64(v[20] as u32, v[21] as u32)),
            remaining_checks: None, // we don't care about this (not Three-Check)
            pockets: None, // we don't care about this (not Crazyhouse)
            castling_rights: Bitboard(join_to_u64(v[22] as u32, v[23] as u32)),
        }, CastlingMode::Standard).unwrap())
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
        match v[0] as u32 {
            0 => {
                let role = v[1] as u32;
                let from = v[2] as u32;
                let capture = if v[3] == 0.0 {
                    None
                } else {
                    Some(v[3] as u32 - 1)
                };
                let to = v[4] as u32;
                let promotion = if v[5] == 0.0 {
                    None
                } else {
                    Some(v[5] as u32 - 1)
                };
                ChessAction(Move::Normal {
                    role: role.try_into().unwrap(),
                    from: from.try_into().unwrap(),
                    capture: capture.map(|x| (x - 1).try_into().unwrap()),
                    to: to.try_into().unwrap(),
                    promotion: promotion.map(|x| (x - 1).try_into().unwrap()),
                })
            }
            1 => {
                let from = v[2] as u32;
                let to = v[4] as u32;
                ChessAction(Move::EnPassant {
                    from: from.try_into().unwrap(),
                    to: to.try_into().unwrap(),
                })
            }
            2 => {
                let king = v[2] as u32;
                let rook = v[4] as u32;
                ChessAction(Move::Castle {
                    king: king.try_into().unwrap(),
                    rook: rook.try_into().unwrap(),
                })
            }
            3 => {
                let role = v[1] as u32;
                let to = v[4] as u32;
                ChessAction(Move::Put {
                    role: role.try_into().unwrap(),
                    to: to.try_into().unwrap(),
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
                        1.0
                    } else {
                        -1.0
                    }
                }
                shakmaty::Outcome::Draw { .. } => 0.0,
            },
            None => 0.0,
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

#[cfg(feature = "dqn")]
fn main() {
    use rurel::strategy::explore::RandomExploration;

    let initial_state = ChessState(Chess::default());

    // Train the agent
    const TRIALS: i32 = 10000;
    let mut trainer = DQNAgentTrainer::<ChessState, 24, 6, 64>::new(0.9, 1e-3);
    for _ in 0..TRIALS {
        let mut agent = ChessAgent(initial_state.clone());
        trainer.train(
            &mut agent,
            &mut ChessTermination,
            &RandomExploration::new(),
        );
        println!("Reward: {}", agent.current_state().reward());
    }

    // Play against the agent
    let mut state = ChessState(Chess::default());


}

#[cfg(not(feature = "dqn"))]
fn main() {
    panic!("Use the 'dqn' feature to run this example");
}

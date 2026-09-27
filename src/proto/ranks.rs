//! Rank-block wire types for the fleet NCCL world (proto v7).
//!
//! The fleet world runs one NCCL rank per GPU. Each NCCL-capable host
//! contributes a *contiguous* block of global ranks, ordered by local GPU
//! index: local GPU `i` of a host with block `base..base+count` is global
//! rank `base + i`. One `agent nccl` process per host drives its whole
//! block.
//!
//! Both types validate on construction *and* on deserialization (serde
//! `try_from`), so a directive that decoded is a directive that makes
//! sense: no empty block, no block overflowing `u32`, no block reaching
//! past the world.

use std::num::NonZeroU32;
use std::ops::Range;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Why a rank block or assignment was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RankError {
    #[error("a rank block must hold at least one rank")]
    EmptyBlock,
    #[error("rank block {base}+{count} overflows the rank space")]
    BlockOverflow { base: u32, count: u32 },
    #[error("rank block {base}..{end} reaches past a world of {world_size}")]
    BlockOutsideWorld {
        base: u32,
        end: u32,
        world_size: u32,
    },
}

/// A host's contiguous block of global NCCL ranks: `base..base + count`,
/// `count >= 1`, never overflowing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RankBlockRepr", into = "RankBlockRepr")]
pub struct RankBlock {
    base: u32,
    count: NonZeroU32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RankBlockRepr {
    base: u32,
    count: u32,
}

impl TryFrom<RankBlockRepr> for RankBlock {
    type Error = RankError;

    fn try_from(repr: RankBlockRepr) -> Result<Self, Self::Error> {
        RankBlock::new(repr.base, repr.count)
    }
}

impl From<RankBlock> for RankBlockRepr {
    fn from(block: RankBlock) -> Self {
        RankBlockRepr {
            base: block.base,
            count: block.count.get(),
        }
    }
}

impl RankBlock {
    pub fn new(base: u32, count: u32) -> Result<Self, RankError> {
        let count = NonZeroU32::new(count).ok_or(RankError::EmptyBlock)?;
        if base.checked_add(count.get()).is_none() {
            return Err(RankError::BlockOverflow {
                base,
                count: count.get(),
            });
        }
        Ok(Self { base, count })
    }

    /// First global rank of the block.
    pub fn base(self) -> u32 {
        self.base
    }

    /// Number of local ranks (= local GPUs driven); always at least 1.
    pub fn count(self) -> u32 {
        self.count.get()
    }

    /// One past the last global rank (cannot overflow: checked at
    /// construction).
    pub fn end(self) -> u32 {
        self.base + self.count.get()
    }

    pub fn ranks(self) -> Range<u32> {
        self.base..self.end()
    }

    pub fn contains(self, rank: u32) -> bool {
        self.ranks().contains(&rank)
    }

    /// Local GPU index of a global rank inside this block.
    pub fn local_index(self, rank: u32) -> Option<u32> {
        self.contains(rank).then(|| rank - self.base)
    }

    /// Global rank of a local GPU index inside this block.
    pub fn global_rank(self, local: u32) -> Option<u32> {
        (local < self.count.get()).then(|| self.base + local)
    }

    /// Whether this block holds global rank 0 — the rendezvous lead and
    /// the only clock that closes a fleet-overlap window.
    pub fn holds_lead(self) -> bool {
        self.base == 0
    }
}

/// One host's position in the fleet world: its rank block and the world
/// size, with `block.end() <= world_size` guaranteed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RankAssignmentRepr", into = "RankAssignmentRepr")]
pub struct RankAssignment {
    block: RankBlock,
    world_size: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RankAssignmentRepr {
    block: RankBlock,
    world_size: u32,
}

impl TryFrom<RankAssignmentRepr> for RankAssignment {
    type Error = RankError;

    fn try_from(repr: RankAssignmentRepr) -> Result<Self, Self::Error> {
        RankAssignment::new(repr.block, repr.world_size)
    }
}

impl From<RankAssignment> for RankAssignmentRepr {
    fn from(assignment: RankAssignment) -> Self {
        RankAssignmentRepr {
            block: assignment.block,
            world_size: assignment.world_size,
        }
    }
}

impl RankAssignment {
    pub fn new(block: RankBlock, world_size: u32) -> Result<Self, RankError> {
        if block.end() > world_size {
            return Err(RankError::BlockOutsideWorld {
                base: block.base(),
                end: block.end(),
                world_size,
            });
        }
        Ok(Self { block, world_size })
    }

    pub fn block(self) -> RankBlock {
        self.block
    }

    /// Total ranks in the fleet world (= GPUs across NCCL-capable hosts);
    /// at least `block.count()`, so never zero.
    pub fn world_size(self) -> u32 {
        self.world_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_reject_zero_counts_and_overflow() {
        assert_eq!(RankBlock::new(0, 0), Err(RankError::EmptyBlock));
        assert_eq!(
            RankBlock::new(u32::MAX, 1),
            Err(RankError::BlockOverflow {
                base: u32::MAX,
                count: 1
            })
        );
        let block = RankBlock::new(u32::MAX - 2, 2).expect("fits exactly");
        assert_eq!(block.end(), u32::MAX);
    }

    #[test]
    fn blocks_map_local_indices_to_global_ranks_and_back() {
        let block = RankBlock::new(8, 4).expect("block");
        assert_eq!(block.ranks(), 8..12);
        assert_eq!(block.count(), 4);
        assert!(!block.holds_lead());
        for local in 0..4 {
            let rank = block.global_rank(local).expect("inside");
            assert_eq!(block.local_index(rank), Some(local));
        }
        assert_eq!(block.global_rank(4), None);
        assert_eq!(block.local_index(7), None);
        assert_eq!(block.local_index(12), None);
        assert!(RankBlock::new(0, 1).expect("lead").holds_lead());
    }

    #[test]
    fn assignments_keep_the_block_inside_the_world() {
        let block = RankBlock::new(4, 4).expect("block");
        assert!(RankAssignment::new(block, 8).is_ok());
        assert_eq!(
            RankAssignment::new(block, 7),
            Err(RankError::BlockOutsideWorld {
                base: 4,
                end: 8,
                world_size: 7
            })
        );
    }

    #[test]
    fn assignments_round_trip_through_json() {
        let assignment =
            RankAssignment::new(RankBlock::new(2, 6).expect("block"), 16).expect("assignment");
        let json = serde_json::to_string(&assignment).expect("serialize");
        assert_eq!(json, r#"{"block":{"base":2,"count":6},"world_size":16}"#);
        let back: RankAssignment = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, assignment);
    }

    #[test]
    fn invalid_wire_documents_are_rejected_on_decode() {
        // Empty block.
        assert!(
            serde_json::from_str::<RankAssignment>(
                r#"{"block":{"base":0,"count":0},"world_size":4}"#
            )
            .is_err()
        );
        // Block past the world.
        assert!(
            serde_json::from_str::<RankAssignment>(
                r#"{"block":{"base":2,"count":4},"world_size":4}"#
            )
            .is_err()
        );
        // Overflowing block.
        assert!(serde_json::from_str::<RankBlock>(r#"{"base":4294967295,"count":1}"#).is_err());
        // Unknown field.
        assert!(serde_json::from_str::<RankBlock>(r#"{"base":0,"count":1,"extra":1}"#).is_err());
    }
}

//! Rank-block wire types for the fleet NCCL world (proto v7).
//!
//! The fleet world runs one NCCL rank per GPU. Each NCCL-capable host
//! contributes a *contiguous* block of global ranks, ordered by local GPU
//! index: local GPU `i` of a host with block `base..base+count` is global
//! rank `base + i`. One `agent nccl` process per host drives its whole
//! block.
//!
//! A block also says which local GPUs its ranks run on: a contiguous run
//! of CUDA device ordinals starting at `first_gpu` (proto v11). The
//! rank-per-GPU world always starts at GPU 0, so local GPU `i` is rank
//! `base + i`; the NIC-forcing world shapes (`rank_per_node`, `per_rail`)
//! give each host a one-rank block on a chosen GPU — rail `r` runs on
//! GPU `r` of every member. `first_gpu` is serde-defaulted to 0 and
//! omitted from the wire when 0, so a rank-per-GPU directive reads as it
//! did before.
//!
//! Both types validate on construction *and* on deserialization (serde
//! `try_from`), so a directive that decoded is a directive that makes
//! sense: no empty block, no block overflowing `u32`, no GPU range
//! overflowing `u32`, no block reaching past the world.

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
    #[error("rank block on gpus {first_gpu}+{count} overflows the device ordinal space")]
    GpuRangeOverflow { first_gpu: u32, count: u32 },
    #[error("rank block {base}..{end} reaches past a world of {world_size}")]
    BlockOutsideWorld {
        base: u32,
        end: u32,
        world_size: u32,
    },
}

/// A host's contiguous block of global NCCL ranks: `base..base + count`,
/// `count >= 1`, never overflowing, running on the local GPUs
/// `first_gpu..first_gpu + count` (also never overflowing): rank
/// `base + i` runs on GPU `first_gpu + i`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RankBlockRepr", into = "RankBlockRepr")]
pub struct RankBlock {
    base: u32,
    count: NonZeroU32,
    first_gpu: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RankBlockRepr {
    base: u32,
    count: u32,
    #[serde(default, skip_serializing_if = "is_zero")]
    first_gpu: u32,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

impl TryFrom<RankBlockRepr> for RankBlock {
    type Error = RankError;

    fn try_from(repr: RankBlockRepr) -> Result<Self, Self::Error> {
        RankBlock::on_gpus(repr.base, repr.count, repr.first_gpu)
    }
}

impl From<RankBlock> for RankBlockRepr {
    fn from(block: RankBlock) -> Self {
        RankBlockRepr {
            base: block.base,
            count: block.count.get(),
            first_gpu: block.first_gpu,
        }
    }
}

impl RankBlock {
    /// A block on local GPUs `0..count` (the rank-per-GPU world).
    pub fn new(base: u32, count: u32) -> Result<Self, RankError> {
        Self::on_gpus(base, count, 0)
    }

    /// A block whose ranks run on local GPUs `first_gpu..first_gpu +
    /// count`.
    pub fn on_gpus(base: u32, count: u32, first_gpu: u32) -> Result<Self, RankError> {
        let count = NonZeroU32::new(count).ok_or(RankError::EmptyBlock)?;
        if base.checked_add(count.get()).is_none() {
            return Err(RankError::BlockOverflow {
                base,
                count: count.get(),
            });
        }
        if first_gpu.checked_add(count.get()).is_none() {
            return Err(RankError::GpuRangeOverflow {
                first_gpu,
                count: count.get(),
            });
        }
        Ok(Self {
            base,
            count,
            first_gpu,
        })
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

    /// Position of a global rank inside this block (0-based). Equal to
    /// the rank's local GPU only when the block starts at GPU 0; use
    /// [`RankBlock::gpu`] for the device.
    pub fn local_index(self, rank: u32) -> Option<u32> {
        self.contains(rank).then(|| rank - self.base)
    }

    /// First local GPU (CUDA device ordinal) the block runs on.
    pub fn first_gpu(self) -> u32 {
        self.first_gpu
    }

    /// The local GPUs (CUDA device ordinals) the block runs on, in rank
    /// order (cannot overflow: checked at construction).
    pub fn gpus(self) -> Range<u32> {
        self.first_gpu..self.first_gpu + self.count.get()
    }

    /// Local GPU (CUDA device ordinal) a global rank of this block runs
    /// on.
    pub fn gpu(self, rank: u32) -> Option<u32> {
        self.local_index(rank).map(|index| self.first_gpu + index)
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
        let locals: Vec<Option<u32>> = block.ranks().map(|rank| block.local_index(rank)).collect();
        assert_eq!(locals, [Some(0), Some(1), Some(2), Some(3)]);
        assert_eq!(block.local_index(7), None);
        assert_eq!(block.local_index(12), None);
        assert!(RankBlock::new(0, 1).expect("lead").holds_lead());
    }

    #[test]
    fn blocks_default_to_gpu_zero() {
        let block = RankBlock::new(8, 4).expect("block");
        assert_eq!(block.first_gpu(), 0);
        assert_eq!(block.gpus(), 0..4);
        let gpus: Vec<Option<u32>> = block.ranks().map(|rank| block.gpu(rank)).collect();
        assert_eq!(gpus, [Some(0), Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn a_block_can_run_on_a_chosen_gpu() {
        // Rail 3 of a per-rail world: one rank per host, on that host's
        // GPU 3.
        let block = RankBlock::on_gpus(2, 1, 3).expect("block");
        assert_eq!(block.ranks(), 2..3);
        assert_eq!(block.local_index(2), Some(0));
        assert_eq!(block.gpu(2), Some(3));
        assert_eq!(block.gpu(3), None);
        assert_eq!(block.gpus(), 3..4);
        assert!(!block.holds_lead());
        assert!(RankBlock::on_gpus(0, 1, 5).expect("lead").holds_lead());
    }

    #[test]
    fn gpu_ranges_cannot_overflow() {
        assert_eq!(
            RankBlock::on_gpus(0, 2, u32::MAX),
            Err(RankError::GpuRangeOverflow {
                first_gpu: u32::MAX,
                count: 2
            })
        );
        assert_eq!(RankBlock::on_gpus(0, 0, 3), Err(RankError::EmptyBlock));
    }

    #[test]
    fn the_gpu_offset_rides_the_wire_only_when_set() {
        let rail = RankAssignment::new(RankBlock::on_gpus(1, 1, 3).expect("block"), 2)
            .expect("assignment");
        let json = serde_json::to_string(&rail).expect("serialize");
        assert_eq!(
            json,
            r#"{"block":{"base":1,"count":1,"first_gpu":3},"world_size":2}"#
        );
        let back: RankAssignment = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, rail);
        // A pre-v11 block without the field decodes onto GPU 0.
        let old: RankBlock = serde_json::from_str(r#"{"base":0,"count":4}"#).expect("decode");
        assert_eq!(old.first_gpu(), 0);
        // Overflowing GPU ranges are rejected on decode, too.
        assert!(
            serde_json::from_str::<RankBlock>(r#"{"base":0,"count":2,"first_gpu":4294967295}"#)
                .is_err()
        );
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

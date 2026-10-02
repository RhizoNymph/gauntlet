//! Rank layout of a fleet NCCL world: each host a contiguous rank block on
//! a contiguous run of its local GPUs (`GpuSpan`), hosts in fleet order.
//! The rank-per-GPU world (`RankLayout::new`) spans every GPU from 0; the
//! NIC-forcing world shapes (`super::shape`) give each host a one-GPU
//! span.
//!
//! Pure: generic over the member type (the orchestrator lays out
//! `Arc<HostSession>`s, tests lay out host names), so the member and its
//! rank block can never drift apart.

use thiserror::Error;

use crate::proto::{RankAssignment, RankBlock, RankError};

/// The local GPUs one member contributes to a world: `count` ranks on
/// GPUs `first..first + count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuSpan {
    pub first: u32,
    pub count: u32,
}

impl GpuSpan {
    /// Every GPU from 0: the rank-per-GPU span of a host with `count`
    /// GPUs.
    pub(crate) fn all(count: u32) -> Self {
        Self { first: 0, count }
    }

    /// The single GPU `gpu`.
    pub(crate) fn one(gpu: u32) -> Self {
        Self {
            first: gpu,
            count: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum LayoutError {
    #[error("the fleet NCCL world would exceed {max} ranks", max = u32::MAX)]
    WorldTooLarge,
    #[error(transparent)]
    Rank(#[from] RankError),
}

/// Where a global rank lives: which member (host) and which local GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RankLocation<'a, M> {
    pub member: &'a M,
    /// Index of the member in fleet order — the rank's *arrival group*:
    /// one process drives every rank of a member, so they share one
    /// launch instant.
    pub member_index: usize,
    /// Local GPU index (= CUDA device ordinal on that host).
    pub gpu: u32,
}

/// A fleet world: members with a non-empty GPU span, each owning the rank
/// block `base..base + span.count` on GPUs `span.first..`, blocks
/// contiguous and ascending in member order, `world_size` = total ranks.
/// Global rank 0 is the first member's first spanned GPU (local GPU 0 in
/// the rank-per-GPU world).
#[derive(Debug, Clone)]
pub(crate) struct RankLayout<M> {
    members: Vec<(M, RankAssignment)>,
    world_size: u32,
}

impl<M> RankLayout<M> {
    /// The world with no members (no NCCL-capable host).
    pub(crate) fn empty() -> Self {
        Self {
            members: Vec::new(),
            world_size: 0,
        }
    }

    /// Lay out `(member, gpu_count)` pairs in the given order. Members
    /// with zero GPUs are excluded (they have no rank to contribute); an
    /// all-zero or empty input yields an empty layout.
    pub(crate) fn new(members: impl IntoIterator<Item = (M, u32)>) -> Result<Self, LayoutError> {
        Self::with_spans(
            members
                .into_iter()
                .map(|(member, gpus)| (member, GpuSpan::all(gpus))),
        )
    }

    /// Lay out `(member, span)` pairs in the given order: each member a
    /// contiguous block of `span.count` ranks on GPUs `span.first..`.
    /// Members with an empty span are excluded.
    pub(crate) fn with_spans(
        members: impl IntoIterator<Item = (M, GpuSpan)>,
    ) -> Result<Self, LayoutError> {
        let mut blocks: Vec<(M, RankBlock)> = Vec::new();
        let mut next: u32 = 0;
        for (member, span) in members {
            if span.count == 0 {
                continue;
            }
            let block =
                RankBlock::on_gpus(next, span.count, span.first).map_err(|error| match error {
                    RankError::BlockOverflow { .. } => LayoutError::WorldTooLarge,
                    other => LayoutError::Rank(other),
                })?;
            next = block.end();
            blocks.push((member, block));
        }
        let world_size = next;
        let members = blocks
            .into_iter()
            .map(|(member, block)| Ok((member, RankAssignment::new(block, world_size)?)))
            .collect::<Result<Vec<_>, LayoutError>>()?;
        Ok(Self {
            members,
            world_size,
        })
    }

    pub(crate) fn world_size(&self) -> u32 {
        self.world_size
    }

    /// Number of members (hosts), i.e. independent arrival groups.
    pub(crate) fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Members in fleet order with their assignments; the first holds
    /// global rank 0.
    pub(crate) fn members(&self) -> &[(M, RankAssignment)] {
        &self.members
    }

    /// Map a global rank to its member and local GPU.
    pub(crate) fn locate(&self, rank: u32) -> Option<RankLocation<'_, M>> {
        // Blocks are contiguous and ascending: the owner is the last block
        // whose base is <= rank.
        let index = self
            .members
            .partition_point(|(_, assignment)| assignment.block().base() <= rank)
            .checked_sub(1)?;
        let (member, assignment) = &self.members[index];
        let gpu = assignment.block().gpu(rank)?;
        Some(RankLocation {
            member,
            member_index: index,
            gpu,
        })
    }

    /// Arrival group of a rank (its member's index), for the barrier-skew
    /// analysis.
    pub(crate) fn arrival_group(&self, rank: u32) -> Option<u32> {
        self.locate(rank)
            .and_then(|location| u32::try_from(location.member_index).ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(hosts: &[(&'static str, u32)]) -> RankLayout<&'static str> {
        RankLayout::new(hosts.iter().copied()).expect("layout")
    }

    fn blocks(layout: &RankLayout<&'static str>) -> Vec<(&'static str, u32, u32)> {
        layout
            .members()
            .iter()
            .map(|(host, assignment)| {
                (*host, assignment.block().base(), assignment.block().count())
            })
            .collect()
    }

    #[test]
    fn spans_put_one_rank_per_host_on_the_chosen_gpu() {
        let layout = RankLayout::with_spans([
            ("n1", GpuSpan::one(3)),
            ("n2", GpuSpan { first: 0, count: 0 }),
            ("n3", GpuSpan::one(3)),
        ])
        .expect("layout");
        assert_eq!(layout.world_size(), 2);
        assert_eq!(blocks(&layout), [("n1", 0, 1), ("n3", 1, 1)]);
        let n3 = layout.locate(1).expect("rank 1");
        assert_eq!((*n3.member, n3.member_index, n3.gpu), ("n3", 1, 3));
        assert_eq!(layout.locate(0).expect("rank 0").gpu, 3);
        assert_eq!(layout.locate(2), None);
        // The directive carries the GPU, so the agent opens the right one.
        assert_eq!(layout.members()[1].1.block().first_gpu(), 3);
    }

    #[test]
    fn rank_per_gpu_spans_match_the_plain_layout() {
        let plain = layout(&[("n1", 2), ("n2", 3)]);
        let spanned = RankLayout::with_spans([("n1", GpuSpan::all(2)), ("n2", GpuSpan::all(3))])
            .expect("layout");
        assert_eq!(blocks(&plain), blocks(&spanned));
        assert_eq!(plain.world_size(), spanned.world_size());
    }

    #[test]
    fn a_span_past_the_device_ordinal_space_is_rejected() {
        let result = RankLayout::with_spans([(
            "n1",
            GpuSpan {
                first: u32::MAX,
                count: 2,
            },
        )]);
        assert!(matches!(
            result,
            Err(LayoutError::Rank(RankError::GpuRangeOverflow { .. }))
        ));
    }

    #[test]
    fn hosts_get_contiguous_blocks_in_fleet_order() {
        let layout = layout(&[("n1", 8), ("n2", 4), ("n3", 8)]);
        assert_eq!(layout.world_size(), 20);
        assert_eq!(blocks(&layout), [("n1", 0, 8), ("n2", 8, 4), ("n3", 12, 8)]);
        for (_, assignment) in layout.members() {
            assert_eq!(assignment.world_size(), 20);
        }
        // Blocks tile 0..world_size with no gap or overlap.
        let mut next = 0;
        for (_, assignment) in layout.members() {
            assert_eq!(assignment.block().base(), next);
            next = assignment.block().end();
        }
        assert_eq!(next, layout.world_size());
    }

    #[test]
    fn zero_gpu_hosts_are_excluded() {
        let layout = layout(&[("cpu0", 0), ("n1", 2), ("cpu1", 0), ("n2", 2)]);
        assert_eq!(blocks(&layout), [("n1", 0, 2), ("n2", 2, 2)]);
        assert_eq!(layout.world_size(), 4);
        assert_eq!(layout.member_count(), 2);
        assert!(layout.members()[0].1.block().holds_lead());
    }

    #[test]
    fn an_empty_or_gpu_less_fleet_has_no_world() {
        let none = layout(&[]);
        assert_eq!(none.member_count(), 0);
        assert_eq!(none.world_size(), 0);
        let cpu_only = layout(&[("a", 0), ("b", 0)]);
        assert_eq!(cpu_only.member_count(), 0);
        assert_eq!(cpu_only.locate(0), None);
    }

    #[test]
    fn a_world_past_u32_is_rejected() {
        let result = RankLayout::new([("a", u32::MAX), ("b", 1)]);
        assert!(matches!(result, Err(LayoutError::WorldTooLarge)));
    }

    #[test]
    fn every_rank_maps_to_its_host_and_local_gpu() {
        let layout = layout(&[("n1", 2), ("n2", 3), ("n3", 1)]);
        let expected = [
            ("n1", 0usize, 0u32),
            ("n1", 0, 1),
            ("n2", 1, 0),
            ("n2", 1, 1),
            ("n2", 1, 2),
            ("n3", 2, 0),
        ];
        for (rank, (host, index, gpu)) in expected.into_iter().enumerate() {
            let location = layout.locate(rank as u32).expect("rank inside the world");
            assert_eq!(*location.member, host, "rank {rank}");
            assert_eq!(location.member_index, index, "rank {rank}");
            assert_eq!(location.gpu, gpu, "rank {rank}");
            assert_eq!(layout.arrival_group(rank as u32), Some(index as u32));
        }
        assert_eq!(layout.locate(6), None);
        assert_eq!(layout.arrival_group(6), None);
    }
}

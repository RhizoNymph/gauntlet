//! Fleet consistency findings: a majority vote per field over every host's
//! inventory fields (`proto::consistency_fields`) and its recorded NCCL NIC
//! fields (`NcclNicSummary::consistency_fields`, from the orchestrator's
//! derivation). Only fields with at least one dissenter are reported.

use std::collections::BTreeMap;

use super::ConsistencyFinding;
use crate::orchestrator::collect::HostObservations;
use crate::proto::{NcclNicSummary, consistency_fields};

pub(super) fn findings(
    observations: &BTreeMap<String, HostObservations>,
) -> BTreeMap<String, ConsistencyFinding> {
    let mut by_field: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for (host, obs) in observations {
        let Some(inventory) = &obs.inventory else {
            continue;
        };
        let nic_fields = obs
            .nccl_nics
            .as_ref()
            .map(NcclNicSummary::consistency_fields)
            .unwrap_or_default();
        for (field, value) in consistency_fields(inventory).into_iter().chain(nic_fields) {
            by_field
                .entry(field)
                .or_default()
                .insert(host.clone(), value);
        }
    }

    let mut findings = BTreeMap::new();
    for (field, values) in by_field {
        let Some(majority_value) = majority(&values) else {
            continue;
        };
        let dissenters: BTreeMap<String, String> = values
            .into_iter()
            .filter(|(_, value)| *value != majority_value)
            .collect();
        if !dissenters.is_empty() {
            findings.insert(
                field,
                ConsistencyFinding {
                    majority_value,
                    dissenters,
                },
            );
        }
    }
    findings
}

/// Most common value; ties broken by the lexicographically smallest value so
/// the result never depends on map iteration luck.
pub(super) fn majority(values: &BTreeMap<String, String>) -> Option<String> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for value in values.values() {
        *counts.entry(value.as_str()).or_default() += 1;
    }
    counts
        .into_iter()
        .min_by(|(a_value, a_count), (b_value, b_count)| {
            b_count.cmp(a_count).then_with(|| a_value.cmp(b_value))
        })
        .map(|(value, _)| value.to_string())
}

//! Consistency-relevant inventory fields: what the fleet majority vote
//! (`report::consistency`) compares across hosts.
//!
//! Values are typed: a field an inventory does not have (no NVIDIA driver,
//! no CUDA toolkit) is `FieldValue::Missing`, never a string that could
//! win a vote. `consistency_fields` is the rendered projection
//! (`Missing` -> `"(absent)"`), which is what reports display.

use std::collections::BTreeMap;
use std::fmt;

use super::InventorySnapshot;

/// How a missing value renders in reports and in `consistency_fields`.
pub const ABSENT: &str = "(absent)";

/// One host's value of one consistency field.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FieldValue {
    Present(String),
    /// The host does not have this at all. Missing never wins a majority
    /// vote over present values, and dissents whenever any host has one.
    Missing,
}

impl FieldValue {
    pub fn from_option(value: Option<String>) -> Self {
        value.map_or(FieldValue::Missing, FieldValue::Present)
    }

    pub fn present(&self) -> Option<&str> {
        match self {
            FieldValue::Present(value) => Some(value),
            FieldValue::Missing => None,
        }
    }

    /// The display form: the value, or `ABSENT`.
    pub fn render(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for FieldValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.present().unwrap_or(ABSENT))
    }
}

/// Typed consistency fields of one inventory, keyed by field name.
pub fn consistency_values(inv: &InventorySnapshot) -> BTreeMap<String, FieldValue> {
    let present = |value: String| FieldValue::Present(value);
    let mut fields = BTreeMap::new();
    fields.insert("kernel".into(), present(inv.kernel.clone()));
    fields.insert("cpu_model".into(), present(inv.cpu_model.clone()));
    // Present-vs-absent is itself a skew signal (one node with a driver the
    // others lack), so Option fields dissent as `Missing` instead of being
    // skipped.
    fields.insert(
        "nvidia_driver".into(),
        FieldValue::from_option(inv.nvidia_driver.clone()),
    );
    fields.insert(
        "cuda_version".into(),
        FieldValue::from_option(inv.cuda_version.clone()),
    );
    fields.insert("gpu_count".into(), present(inv.gpus.len().to_string()));
    if !inv.gpus.is_empty() {
        for (lib, available) in &inv.gpu_libs {
            fields.insert(
                format!("lib:{lib}"),
                present(if *available { "present" } else { "absent" }.to_string()),
            );
        }
    }
    if let Some(gpu) = inv.gpus.first() {
        fields.insert("gpu_model".into(), present(gpu.name.clone()));
        fields.insert("vbios".into(), present(gpu.vbios.clone()));
    }
    for nic in &inv.nics {
        fields.insert(format!("mtu:{}", nic.name), present(nic.mtu.to_string()));
    }
    fields
}

/// Consistency-relevant fields extracted from an inventory, keyed by field
/// name, rendered for display (`Missing` as `ABSENT`). Fleet analysis
/// votes on the typed `consistency_values`.
pub fn consistency_fields(inv: &InventorySnapshot) -> BTreeMap<String, String> {
    consistency_values(inv)
        .into_iter()
        .map(|(field, value)| (field, value.render()))
        .collect()
}

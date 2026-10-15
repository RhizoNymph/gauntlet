//! PCI function addresses (`domain:bus:device.function`), typed.
//!
//! Shared by the agent's probes (nvidia-smi bus ids, the sysfs PCI walk)
//! and the wire (`GpuDriverState::DriverUnavailable` lists the NVIDIA GPUs
//! found on the bus by address). On the wire an address is its canonical
//! string (`0000:01:00.0`); decoding validates it.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A PCI address, normalized so that nvidia-smi's two spellings
/// (`00000000:01:00.0` from both queries on most drivers, `0000:01:00.0`
/// elsewhere) compare equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct PciBusId {
    domain: u32,
    bus: u8,
    device: u8,
    function: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PciBusIdError {
    #[error("pci bus id {raw:?} is not domain:bus:device.function")]
    Shape { raw: String },
    #[error("pci bus id {raw:?} has a non-hex {part} component")]
    Component { raw: String, part: &'static str },
}

impl FromStr for PciBusId {
    type Err = PciBusIdError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let trimmed = raw.trim();
        let shape = || PciBusIdError::Shape {
            raw: raw.to_string(),
        };
        let mut colon_parts = trimmed.split(':');
        let (Some(domain), Some(bus), Some(slot), None) = (
            colon_parts.next(),
            colon_parts.next(),
            colon_parts.next(),
            colon_parts.next(),
        ) else {
            return Err(shape());
        };
        let (device, function) = slot.split_once('.').ok_or_else(shape)?;
        let component = |text: &str, part: &'static str| {
            (!text.is_empty())
                .then_some(text)
                .and_then(|text| u32::from_str_radix(text, 16).ok())
                .ok_or_else(|| PciBusIdError::Component {
                    raw: raw.to_string(),
                    part,
                })
        };
        let narrow = |value: u32, part: &'static str| {
            u8::try_from(value).map_err(|_| PciBusIdError::Component {
                raw: raw.to_string(),
                part,
            })
        };
        Ok(Self {
            domain: component(domain, "domain")?,
            bus: narrow(component(bus, "bus")?, "bus")?,
            device: narrow(component(device, "device")?, "device")?,
            function: narrow(component(function, "function")?, "function")?,
        })
    }
}

impl fmt::Display for PciBusId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

impl TryFrom<String> for PciBusId {
    type Error = PciBusIdError;

    fn try_from(raw: String) -> Result<Self, Self::Error> {
        raw.parse()
    }
}

impl From<PciBusId> for String {
    fn from(id: PciBusId) -> Self {
        id.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_round_trip_as_canonical_strings() {
        let id: PciBusId = "00000000:01:00.0".parse().expect("parse");
        let json = serde_json::to_string(&id).expect("serialize");
        assert_eq!(json, r#""0000:01:00.0""#);
        assert_eq!(serde_json::from_str::<PciBusId>(&json).expect("decode"), id);
        assert!(serde_json::from_str::<PciBusId>(r#""not-an-address""#).is_err());
    }
}

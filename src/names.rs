//! Spelling tables for small closed enums that users name on the command
//! line and in config (`Phase`, `NetStep`): one table of accepted
//! spellings drives parsing, help text and serde, so the CLI, the config
//! file and the help can never disagree.

use std::fmt;

/// A closed enum with a canonical name per variant and a table of every
/// accepted spelling (canonical names plus aliases).
pub trait NameTable: Copy + PartialEq + 'static {
    /// What the values are called in messages ("phase", "network step").
    const KIND: &'static str;
    /// Every variant, in run order.
    const VARIANTS: &'static [Self];
    /// Every accepted spelling. Must contain each variant's `name`.
    const SPELLINGS: &'static [(&'static str, Self)];

    /// Canonical spelling.
    fn name(self) -> &'static str;
}

/// Look `s` up in `T::SPELLINGS`.
pub fn parse<T: NameTable>(s: &str) -> Option<T> {
    T::SPELLINGS
        .iter()
        .find(|(alias, _)| *alias == s)
        .map(|(_, value)| *value)
}

/// Run-order canonical names with their aliases in parentheses, e.g.
/// `inventory, cpu_mem (cpu), gpu`.
pub fn help_list<T: NameTable>() -> String {
    T::VARIANTS
        .iter()
        .map(|value| {
            let aliases: Vec<&str> = T::SPELLINGS
                .iter()
                .filter(|(alias, target)| target == value && *alias != value.name())
                .map(|(alias, _)| *alias)
                .collect();
            if aliases.is_empty() {
                value.name().to_string()
            } else {
                format!("{} ({})", value.name(), aliases.join(", "))
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// A spelling that is not in the table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownName {
    pub kind: &'static str,
    pub name: String,
    /// `help_list` of the table, for the message.
    pub expected: String,
}

impl fmt::Display for UnknownName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown {} {:?} (expected one of: {})",
            self.kind, self.name, self.expected
        )
    }
}

impl std::error::Error for UnknownName {}

/// `parse`, with a descriptive error.
pub fn parse_named<T: NameTable>(s: &str) -> Result<T, UnknownName> {
    parse(s).ok_or_else(|| UnknownName {
        kind: T::KIND,
        name: s.to_string(),
        expected: help_list::<T>(),
    })
}

/// Implement `Serialize` (canonical name) and `Deserialize` (any spelling
/// in the table) for a `NameTable` type.
#[macro_export]
macro_rules! serde_via_name_table {
    ($ty:ty) => {
        impl serde::Serialize for $ty {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str($crate::names::NameTable::name(*self))
            }
        }

        impl<'de> serde::Deserialize<'de> for $ty {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let raw = String::deserialize(deserializer)?;
                $crate::names::parse_named::<$ty>(&raw).map_err(serde::de::Error::custom)
            }
        }
    };
}

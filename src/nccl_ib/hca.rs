//! NCCL's `NCCL_IB_HCA` device filter, modelled on NCCL's own parser
//! (`parseStringList` / `matchIfList` in `src/misc/utils.cc`, applied in
//! the IB transport's device scan).
//!
//! Grammar, exactly as NCCL reads it:
//! - An optional leading `^` inverts the whole list (exclude).
//! - Then an optional `=` switches every entry from prefix to exact name
//!   matching. Order matters: `^=mlx5_0` is "exclude exactly mlx5_0";
//!   `=^mlx5_0` is an exact include of a device literally named
//!   `^mlx5_0`.
//! - The rest is a comma-separated list of `name` or `name:port`. Empty
//!   entries (`a,,b`, a trailing comma) are skipped. Nothing is trimmed:
//!   `mlx5_0, mlx5_1` has an entry " mlx5_1" that matches no device.
//! - A port is read with C `atoi` semantics: leading whitespace, optional
//!   sign, leading digits, anything else ignored, no digits = 0. Port -1
//!   means "any port" (NCCL's internal sentinel); any other number must
//!   equal the device port, so `mlx5_0:0` and `mlx5_0:x` match nothing.
//! - A name with an empty prefix before `:` (`:1`) is dropped.
//! - An empty list matches every device; with `^` it therefore excludes
//!   every device (NCCL computes `match(list) XOR exclude`).
//!
//! Not modelled: NCCL's 32-entry list cap (`MAX_IB_DEVS`) and the 64-byte
//! prefix cap — both far beyond any real host.

/// Which port(s) an entry names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcaPort {
    /// No `:port` given (or NCCL's -1 sentinel).
    Any,
    /// `:N` as `atoi` read it; only equals a real port when N >= 1.
    Number(i64),
}

/// One `name[:port]` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HcaEntry {
    pub name: String,
    pub port: HcaPort,
}

/// A parsed `NCCL_IB_HCA` value. Parsing never fails: NCCL accepts any
/// string, so every string has a meaning here too.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HcaFilter {
    exclude: bool,
    exact: bool,
    entries: Vec<HcaEntry>,
}

impl HcaFilter {
    /// `NCCL_IB_HCA` unset: every device and port matches.
    pub fn unset() -> Self {
        Self::default()
    }

    pub fn parse(value: &str) -> Self {
        let (exclude, rest) = match value.strip_prefix('^') {
            Some(rest) => (true, rest),
            None => (false, value),
        };
        let (exact, rest) = match rest.strip_prefix('=') {
            Some(rest) => (true, rest),
            None => (false, rest),
        };
        let entries = rest
            .split(',')
            .filter_map(|entry| {
                let (name, port) = match entry.split_once(':') {
                    Some((name, port)) => (name, parse_port(port)),
                    None => (entry, HcaPort::Any),
                };
                (!name.is_empty()).then(|| HcaEntry {
                    name: name.to_string(),
                    port,
                })
            })
            .collect();
        Self {
            exclude,
            exact,
            entries,
        }
    }

    pub fn exclude(&self) -> bool {
        self.exclude
    }

    pub fn exact(&self) -> bool {
        self.exact
    }

    pub fn entries(&self) -> &[HcaEntry] {
        &self.entries
    }

    /// Would NCCL keep `device:port` under this filter?
    pub fn matches(&self, device: &str, port: u32) -> bool {
        let listed = self.entries.is_empty()
            || self.entries.iter().any(|entry| {
                let name_matches = if self.exact {
                    device == entry.name
                } else {
                    device.starts_with(&entry.name)
                };
                let port_matches = match entry.port {
                    HcaPort::Any => true,
                    HcaPort::Number(number) => number == i64::from(port),
                };
                name_matches && port_matches
            });
        listed != self.exclude
    }
}

/// C `atoi` over the text after `:` (NCCL stops the entry at the next
/// comma, which `split(',')` already did). -1 is NCCL's "any port".
fn parse_port(text: &str) -> HcaPort {
    let text = text.trim_start();
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let magnitude = digits
        .bytes()
        .take_while(u8::is_ascii_digit)
        .fold(0i64, |acc, digit| {
            acc.saturating_mul(10)
                .saturating_add(i64::from(digit - b'0'))
        });
    match if negative { -magnitude } else { magnitude } {
        -1 => HcaPort::Any,
        number => HcaPort::Number(number),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEVICES: [(&str, u32); 6] = [
        ("mlx5_0", 1),
        ("mlx5_1", 1),
        ("mlx5_1", 2),
        ("mlx5_10", 1),
        ("mlx5_bond_0", 1),
        ("irdma0", 1),
    ];

    fn selected(value: Option<&str>) -> Vec<String> {
        let filter = value.map_or_else(HcaFilter::unset, HcaFilter::parse);
        DEVICES
            .iter()
            .filter(|(device, port)| filter.matches(device, *port))
            .map(|(device, port)| format!("{device}:{port}"))
            .collect()
    }

    #[test]
    fn unset_matches_everything() {
        assert_eq!(selected(None).len(), DEVICES.len());
        // An empty value behaves like an empty list.
        assert_eq!(selected(Some("")).len(), DEVICES.len());
    }

    #[test]
    fn plain_names_are_prefixes() {
        // mlx5_1 also matches mlx5_10 (and both of mlx5_1's ports).
        assert_eq!(
            selected(Some("mlx5_1")),
            ["mlx5_1:1", "mlx5_1:2", "mlx5_10:1"]
        );
        assert_eq!(
            selected(Some("mlx5")),
            [
                "mlx5_0:1",
                "mlx5_1:1",
                "mlx5_1:2",
                "mlx5_10:1",
                "mlx5_bond_0:1"
            ]
        );
    }

    #[test]
    fn comma_lists_union() {
        assert_eq!(selected(Some("mlx5_0,irdma")), ["mlx5_0:1", "irdma0:1"]);
        // Empty entries are skipped.
        assert_eq!(selected(Some(",mlx5_0,,")), ["mlx5_0:1"]);
    }

    #[test]
    fn equals_prefix_makes_every_entry_exact() {
        assert_eq!(selected(Some("=mlx5_1")), ["mlx5_1:1", "mlx5_1:2"]);
        assert_eq!(selected(Some("=mlx5")), Vec::<String>::new());
        assert_eq!(selected(Some("=mlx5_0,mlx5_10")), ["mlx5_0:1", "mlx5_10:1"]);
    }

    #[test]
    fn caret_excludes() {
        assert_eq!(
            selected(Some("^mlx5_1")),
            ["mlx5_0:1", "mlx5_bond_0:1", "irdma0:1"]
        );
        assert_eq!(
            selected(Some("^=mlx5_1")),
            ["mlx5_0:1", "mlx5_10:1", "mlx5_bond_0:1", "irdma0:1"]
        );
        // "^" alone: empty list matches all, inverted -> nothing.
        assert_eq!(selected(Some("^")), Vec::<String>::new());
        // "=^..." is an exact include of a name starting with '^'.
        assert_eq!(selected(Some("=^mlx5_0")), Vec::<String>::new());
        let filter = HcaFilter::parse("=^mlx5_0");
        assert!(!filter.exclude() && filter.exact());
        assert_eq!(filter.entries()[0].name, "^mlx5_0");
    }

    #[test]
    fn dev_port_form_pins_the_port() {
        assert_eq!(selected(Some("mlx5_1:2")), ["mlx5_1:2"]);
        assert_eq!(
            selected(Some("=mlx5_1:1,mlx5_0:1")),
            ["mlx5_0:1", "mlx5_1:1"]
        );
        assert_eq!(
            selected(Some("^mlx5_1:2")),
            [
                "mlx5_0:1",
                "mlx5_1:1",
                "mlx5_10:1",
                "mlx5_bond_0:1",
                "irdma0:1"
            ]
        );
        // atoi: no digits -> 0, trailing junk ignored, -1 -> any port.
        assert_eq!(selected(Some("mlx5_1:x")), Vec::<String>::new());
        assert_eq!(selected(Some("mlx5_1:0")), Vec::<String>::new());
        assert_eq!(selected(Some("mlx5_1:2abc")), ["mlx5_1:2"]);
        assert_eq!(selected(Some("=mlx5_1:-1")), ["mlx5_1:1", "mlx5_1:2"]);
        assert_eq!(selected(Some("mlx5_0: 1")), ["mlx5_0:1"]);
        // ":1" has no name and is dropped, leaving an empty list.
        assert_eq!(selected(Some(":1")).len(), DEVICES.len());
    }

    #[test]
    fn nothing_is_trimmed() {
        assert_eq!(selected(Some("mlx5_0, mlx5_10")), ["mlx5_0:1"]);
    }

    #[test]
    fn parsed_entries_are_inspectable() {
        let filter = HcaFilter::parse("^=mlx5_0:1,mlx5_3");
        assert!(filter.exclude());
        assert!(filter.exact());
        assert_eq!(
            filter.entries(),
            [
                HcaEntry {
                    name: "mlx5_0".into(),
                    port: HcaPort::Number(1)
                },
                HcaEntry {
                    name: "mlx5_3".into(),
                    port: HcaPort::Any
                },
            ]
        );
    }
}

//! Parsing an order file and applying it to a tensor list.

use spm_import::{ImportError, Tensor};
use std::path::Path;

/// Section header starting the region swept and rewound.
const ROTATING: &str = "[rotating]";
/// Section header starting the region read once into RAM.
const RESIDENT: &str = "[resident]";

/// Tensor names in consumption order, split by how they are read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Order {
    /// Swept once per operation and rewound. Must be emitted first:
    /// `rewind` returns to the start of the stream, so the region
    /// that gets rewound has to begin there.
    pub rotating: Vec<String>,
    /// Read once and kept in RAM. Small state, which docs/plan.md
    /// section 3 puts in ordinary memory on purpose.
    pub resident: Vec<String>,
}

/// Parses an order file: `[rotating]` and `[resident]` sections, one
/// tensor name per line, `#` comments and blanks ignored.
///
/// # Errors
/// Returns [`ImportError::BadLine`] for a name outside any section.
pub fn parse_order(text: &str) -> Result<Order, ImportError> {
    let mut order = Order::default();
    let mut section = None;
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match line {
            ROTATING => section = Some(true),
            RESIDENT => section = Some(false),
            name => match section {
                Some(true) => order.rotating.push(name.to_string()),
                Some(false) => order.resident.push(name.to_string()),
                None => {
                    return Err(ImportError::BadLine {
                        line: index + 1,
                        why: "name before any [rotating] or [resident] section",
                    });
                }
            },
        }
    }
    Ok(order)
}

/// Reorders `tensors` to match `order`, rotating region first.
///
/// Both directions are checked. A name in the order file but not in
/// the checkpoint means the layout is stale; a tensor in the
/// checkpoint but not in the order file would be emitted somewhere
/// arbitrary. Silence on either is how a layout drifts away from the
/// model it describes.
///
/// # Errors
/// Returns [`ImportError::BadLayout`] naming the first mismatch.
pub fn reorder(tensors: &[Tensor], order: &Order) -> Result<Vec<Tensor>, ImportError> {
    let wanted: Vec<&String> = order.rotating.iter().chain(&order.resident).collect();
    let mut out = Vec::with_capacity(wanted.len());
    for name in &wanted {
        let found = tensors.iter().find(|t| &&t.name == name);
        out.push(found.cloned().ok_or_else(|| ImportError::BadLayout {
            name: (*name).clone(),
            why: "named in the order file but not in the checkpoint",
        })?);
    }
    for tensor in tensors {
        if !wanted.iter().any(|n| **n == tensor.name) {
            return Err(ImportError::BadLayout {
                name: tensor.name.clone(),
                why: "in the checkpoint but not named in the order file",
            });
        }
    }
    Ok(out)
}

/// Reads an order file and applies it, returning the ordered tensors
/// and how many of them form the rotating region.
///
/// With no order file the tensors keep the manifest's order and the
/// rotating region is empty. That is honest rather than convenient:
/// the extractor sorts alphabetically, and a caller who did not supply
/// an order should not be told they have a rotating region they do
/// not have.
///
/// # Errors
/// Returns [`ImportError`] if the file cannot be read, is malformed,
/// or disagrees with the checkpoint.
pub fn apply_order(
    tensors: Vec<Tensor>,
    path: Option<&Path>,
) -> Result<(Vec<Tensor>, usize), ImportError> {
    let Some(path) = path else {
        return Ok((tensors, 0));
    };
    let text = std::fs::read_to_string(path).map_err(|e| ImportError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    let order = parse_order(&text)?;
    let rotating = order.rotating.len();
    Ok((reorder(&tensors, &order)?, rotating))
}

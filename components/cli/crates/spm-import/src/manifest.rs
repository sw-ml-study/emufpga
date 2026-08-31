//! Reading the extractor's `manifest.tsv`.

use std::fmt;

/// One tensor as the extractor described it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// Name as it appeared in the checkpoint's state dict.
    pub name: String,
    /// Dimensions, outermost first.
    pub shape: Vec<u32>,
    /// File holding its little-endian `f32` bytes.
    pub blob: String,
}

impl Tensor {
    /// This tensor's shape as a stream's `(rows, cols)`.
    ///
    /// A 2-D tensor `(out, in)` maps to `rows = out`, `cols = in`,
    /// matching how a weight matrix multiplies an activation vector.
    ///
    /// A 1-D tensor of `n` becomes `rows = n`, `cols = 1` -- a single
    /// column. Not arbitrary: the stream is column-major, so one
    /// column means the weights stream in exactly their natural order,
    /// which is what a bias or an init vector wants.
    ///
    /// More than two dimensions flattens to
    /// `(product of leading dims, last dim)`. TRM has none, so this is
    /// stated rather than exercised.
    #[must_use]
    pub fn stream_shape(&self) -> (u32, u32) {
        match self.shape.as_slice() {
            [] => (0, 0),
            [n] => (*n, 1),
            [.., last] => (self.shape[..self.shape.len() - 1].iter().product(), *last),
        }
    }
}

/// A checkpoint that could not be imported.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImportError {
    /// A manifest line did not have the expected fields.
    BadLine {
        /// 1-based line number.
        line: usize,
        /// What was wrong.
        why: &'static str,
    },
    /// A blob was not the size its shape implies.
    ///
    /// Caught before anything is written: a short blob would otherwise
    /// produce a `.spm` whose descriptors promise more weights than
    /// the payload holds, and the reader would only notice at the end.
    BlobSize {
        /// Which tensor.
        name: String,
        /// Bytes the shape implies.
        expected: usize,
        /// Bytes found.
        found: usize,
    },
    /// A file could not be read or written.
    Io {
        /// What was being accessed.
        path: String,
        /// The underlying message.
        detail: String,
    },
    /// The order file and the checkpoint disagree.
    BadLayout {
        /// The tensor at fault.
        name: String,
        /// How they disagree.
        why: &'static str,
    },
    /// The `.spm` writer rejected the file.
    Format {
        /// The underlying message.
        detail: String,
    },
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadLine { line, why } => write!(f, "manifest line {line}: {why}"),
            Self::BlobSize {
                name,
                expected,
                found,
            } => write!(
                f,
                "{name}: shape implies {expected} bytes, blob has {found}"
            ),
            Self::Io { path, detail } => write!(f, "{path}: {detail}"),
            Self::BadLayout { name, why } => write!(f, "layout: {name} is {why}"),
            Self::Format { detail } => write!(f, "cannot build .spm: {detail}"),
        }
    }
}

impl std::error::Error for ImportError {}

/// Parses a `manifest.tsv`.
///
/// Blank lines and `#` comments are ignored, so the extractor can
/// label its columns.
///
/// # Errors
/// Returns [`ImportError::BadLine`] for a line with too few fields or
/// an unparseable dimension.
pub fn parse_manifest(text: &str) -> Result<Vec<Tensor>, ImportError> {
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        out.push(parse_line(line, index + 1)?);
    }
    Ok(out)
}

/// Parses one `name<TAB>shape<TAB>dtype<TAB>blob<TAB>elements` line.
fn parse_line(line: &str, number: usize) -> Result<Tensor, ImportError> {
    let fields: Vec<&str> = line.split('\t').collect();
    let bad = |why| ImportError::BadLine { line: number, why };
    if fields.len() < 4 {
        return Err(bad("expected name, shape, dtype, blob"));
    }
    let shape = fields[1]
        .split(',')
        .map(|d| d.trim().parse::<u32>().map_err(|_| bad("bad dimension")))
        .collect::<Result<Vec<u32>, _>>()?;
    if shape.is_empty() || shape.contains(&0) {
        return Err(bad("shape must be non-empty and nonzero"));
    }
    Ok(Tensor {
        name: fields[0].to_string(),
        shape,
        blob: fields[3].to_string(),
    })
}

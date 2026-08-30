//! Reading a dense f32 matrix from text.

use std::fmt;

/// A dense matrix, row-major, as written by a human.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    /// Rows, M. Inferred from the line count.
    pub rows: usize,
    /// Columns, N. Inferred from the first line's value count.
    pub cols: usize,
    /// Values, row-major.
    pub values: Vec<f32>,
}

/// A matrix text file that could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatrixError {
    /// A token was not a float.
    BadNumber {
        /// 1-based line number.
        line: usize,
        /// The offending token.
        found: String,
    },
    /// A row had a different width from the first row.
    RaggedRow {
        /// 1-based line number.
        line: usize,
        /// Columns the first row established.
        expected: usize,
        /// Columns this row holds.
        found: usize,
    },
    /// The file held no rows.
    Empty,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadNumber { line, found } => {
                write!(f, "line {line}: {found:?} is not a number")
            }
            Self::RaggedRow {
                line,
                expected,
                found,
            } => write!(
                f,
                "line {line}: row has {found} columns, expected {expected}"
            ),
            Self::Empty => write!(f, "matrix file has no rows"),
        }
    }
}

impl std::error::Error for MatrixError {}

/// Parses whitespace-separated floats, one matrix row per line.
///
/// Blank lines and lines starting with `#` are ignored, so a matrix
/// file can carry a comment saying where it came from. Shape is
/// inferred rather than passed as flags: the file already knows its
/// own dimensions, and a `--rows` flag that disagrees with the file is
/// one more thing to get wrong.
///
/// # Errors
/// Returns [`MatrixError`] if the file is empty, a row is ragged, or a
/// token is not a float.
pub fn parse_matrix(text: &str) -> Result<Matrix, MatrixError> {
    let mut values = Vec::new();
    let mut rows = 0usize;
    let mut cols = 0usize;
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let width = absorb_row(line, index + 1, &mut values)?;
        if rows == 0 {
            cols = width;
        } else if width != cols {
            return Err(MatrixError::RaggedRow {
                line: index + 1,
                expected: cols,
                found: width,
            });
        }
        rows += 1;
    }
    (rows > 0).then_some(()).ok_or(MatrixError::Empty)?;
    Ok(Matrix { rows, cols, values })
}

/// Parses one row's floats, returning its width.
fn absorb_row(line: &str, number: usize, into: &mut Vec<f32>) -> Result<usize, MatrixError> {
    let mut width = 0;
    for token in line.split_whitespace() {
        into.push(token.parse().map_err(|_| MatrixError::BadNumber {
            line: number,
            found: token.into(),
        })?);
        width += 1;
    }
    Ok(width)
}

/// Reorders row-major values into column-major stream order.
///
/// This is the only place the transposition happens. Input is
/// row-major because that is how a human writes a matrix; output is
/// column-major because that is the order the engine consumes
/// (docs/spm-format.md).
#[must_use]
pub fn to_stream_order(matrix: &Matrix) -> Vec<f32> {
    let mut out = Vec::with_capacity(matrix.values.len());
    for col in 0..matrix.cols {
        for row in 0..matrix.rows {
            out.push(matrix.values[row * matrix.cols + col]);
        }
    }
    out
}

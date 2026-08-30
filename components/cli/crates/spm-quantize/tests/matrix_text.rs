//! Reading the text matrix format, including what it refuses.

use spm_quantize::{MatrixError, parse_matrix};

#[test]
fn shape_is_inferred_and_comments_are_ignored() {
    let matrix = parse_matrix("# from somewhere\n\n1 2 3\n4 5 6\n").expect("parse");
    assert_eq!((matrix.rows, matrix.cols), (2, 3));
    assert_eq!(matrix.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn a_ragged_row_names_its_line() {
    assert_eq!(
        parse_matrix("1 2 3\n4 5\n"),
        Err(MatrixError::RaggedRow {
            line: 2,
            expected: 3,
            found: 2
        })
    );
}

#[test]
fn a_bad_token_names_its_line_and_itself() {
    assert_eq!(
        parse_matrix("1 2\n3 oops\n"),
        Err(MatrixError::BadNumber {
            line: 2,
            found: "oops".into()
        })
    );
}

#[test]
fn an_empty_file_is_refused() {
    assert_eq!(parse_matrix(""), Err(MatrixError::Empty));
    assert_eq!(parse_matrix("# only a comment\n"), Err(MatrixError::Empty));
}

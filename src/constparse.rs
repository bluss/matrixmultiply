// Copyright (c) 2021 DutchGhost
// Copyright (c) 2021 matrixmultiply authors
//
// Incorporpated in matrixmultiply under these terms, see main license files.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Clone, Copy)]
pub(crate) enum ParseIntError {
    InvalidDigit,
}

const fn parse_byte(b: u8, pow10: usize) -> Result<usize, ParseIntError> {
    let r = b.wrapping_sub(48);

    if r > 9 {
        Err(ParseIntError::InvalidDigit)
    } else {
        Ok((r as usize) * pow10)
    }
}

pub(crate) const POW10: [usize; 20] = {
    let mut array = [0; 20];
    let mut current = 1;

    let mut index = 20;

    loop {
        index -= 1;
        array[index] = current;

        if index == 0 { break }

        current *= 10;

    }

    array

};

/// Parse the input to integer; or otherwise cause
/// a const error, an "unwarp" in space and time.
pub(crate) const fn parse_unwarp(b: &str) -> usize {
    match parse(b) {
        Ok(t) => t,
        res @ Err(_) => {
            [0, /* const error: failed to parse environment variable */][res.is_err() as usize]
        }
    }
}

/// Parse the input to usize
pub(crate) const fn parse(b: &str) -> Result<usize, ParseIntError> {
    let bytes = b.as_bytes();

    let mut result: usize = 0;

    let len = bytes.len();

    // Start at the correct index of the table,
    // (skip the power's that are too large)
    let mut index_const_table = POW10.len().wrapping_sub(len);
    let mut index = 0;

    while index < b.len() {
        let a = bytes[index];
        let p = POW10[index_const_table];

        let r = match parse_byte(a, p) {
            Err(e) => return Err(e),
            Ok(d) => d,
        };

        result = result.wrapping_add(r);

        index += 1;
        index_const_table += 1;
    }

    Ok(result)
}

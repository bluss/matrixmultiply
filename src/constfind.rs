//! Copyright 2026 Ulrik Sverdrup "bluss"

pub(crate) const fn slice_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] { return false; }
        i += 1;
    }
    true
}

const fn find_byte(text: &[u8], byte: u8) -> Option<usize> {
    let mut j = 0;
    while j < text.len() && text[j] != byte {
        j += 1;
    }
    if j == text.len() { None } else { Some(j) }
}

/// Search for exact word match in string of comma separated words
///
/// Example "avx2,fma", "avx2" => true; "avx2,fma", "avx" => false
pub(crate) const fn comma_separated_contains(text: &str, word: &str) -> bool {
    let mut text = text.as_bytes();
    loop {
        let next_comma = find_byte(text, b',');
        let word_end = match next_comma { Some(x) => x, None => text.len() };
        let (this_word, _) = text.split_at(word_end);
        if slice_eq(this_word, word.as_bytes()) {
            return true;
        }

        // take next segment
        if let None = next_comma {
            return false;
        }
        let (_, tail) = text.split_at(word_end + 1);
        text = tail;
    }
}

#[test]
fn test_find_byte() {
    assert_eq!(find_byte(b"abc", b'z'), None);
    assert_eq!(find_byte(b"abc", b'b'), Some(1));
}

#[test]
fn test_comma_separated_contains() {
    assert!(comma_separated_contains("abc,xyz", "abc"));
    assert!(comma_separated_contains("abc,xyz", "xyz"));
    assert!(!comma_separated_contains("abc,xyz", "abc,"));
    assert!(!comma_separated_contains("avx2,fma", "avx"));
}



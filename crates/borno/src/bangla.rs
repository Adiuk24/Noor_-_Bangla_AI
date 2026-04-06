use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

const BENGALI_START: u32 = 0x0980;
const BENGALI_END: u32 = 0x09FF;

// Bengali anusvara (ং) and visarga (ঃ) are spacing marks (Mc) that Unicode
// attaches to a preceding base letter. In Bangla linguistics they are
// independent aksharas, so we treat them as grapheme-cluster breakers.
const BENGALI_ANUSVARA: char = '\u{0982}'; // ং
const BENGALI_VISARGA: char = '\u{0983}';  // ঃ

fn is_bengali_standalone_mark(c: char) -> bool {
    c == BENGALI_ANUSVARA || c == BENGALI_VISARGA
}

pub fn is_bengali_char(c: char) -> bool {
    let cp = c as u32;
    (BENGALI_START..=BENGALI_END).contains(&cp)
}

pub fn normalize(input: &str) -> String {
    input.nfc().collect()
}

/// Return grapheme clusters for Bangla text, treating anusvara (ং) and
/// visarga (ঃ) as standalone aksharas rather than attaching them to the
/// preceding base (which is what the Unicode standard does but is wrong for
/// Bangla linguistic segmentation).
pub fn grapheme_clusters(input: &str) -> Vec<&str> {
    // We walk through standard Unicode grapheme clusters and then split any
    // cluster that contains an anusvara/visarga that is not the first character
    // (i.e. it attached to a preceding base by the Unicode algorithm).
    let mut result: Vec<&str> = Vec::new();

    for cluster in input.graphemes(true) {
        // Check if this cluster contains a standalone mark that should be split off.
        // Strategy: find the byte offset of the first anusvara/visarga inside
        // the cluster. If it is not the very first char, split there.
        let mut split_byte: Option<usize> = None;
        let mut byte_offset = 0usize;
        let mut first = true;
        for c in cluster.chars() {
            if !first && is_bengali_standalone_mark(c) {
                split_byte = Some(byte_offset);
                break;
            }
            byte_offset += c.len_utf8();
            first = false;
        }

        if let Some(split) = split_byte {
            // Split cluster into two parts: [0..split] and [split..]
            // The mark part may itself need further splitting if the tail has more chars.
            let head = &cluster[..split];
            let tail = &cluster[split..];
            if !head.is_empty() {
                result.push(head);
            }
            // The tail is typically just the standalone mark(s); recurse to handle
            // edge cases where tail still has attached content.
            // For the common cases we just push the tail as-is.
            if !tail.is_empty() {
                result.push(tail);
            }
        } else {
            result.push(cluster);
        }
    }

    result
}

pub fn normalize_and_segment(input: &str) -> Vec<String> {
    let normalized = normalize(input);
    grapheme_clusters(&normalized)
        .into_iter()
        .map(String::from)
        .collect()
}

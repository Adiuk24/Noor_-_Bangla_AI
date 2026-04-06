/// Total vocabulary size.
pub const VOCAB_SIZE: usize = 64_000;

/// Byte fallback tokens occupy IDs 0-255.
pub const BYTE_FALLBACK_START: u32 = 0;
pub const BYTE_FALLBACK_END: u32 = 255;

/// Special token IDs.
pub const BOS_ID: u32 = 256;
pub const EOS_ID: u32 = 257;
pub const PAD_ID: u32 = 258;
pub const UNK_ID: u32 = 259;
pub const USER_ID: u32 = 260;
pub const ASSISTANT_ID: u32 = 261;
pub const SYSTEM_ID: u32 = 262;
pub const TOOL_CALL_ID: u32 = 263;
pub const TOOL_RESULT_ID: u32 = 264;
pub const THINK_ID: u32 = 265;
pub const THINK_END_ID: u32 = 266;
pub const MEMORY_ID: u32 = 267;
pub const MEMORY_END_ID: u32 = 268;
pub const CODE_ID: u32 = 269;
pub const CODE_END_ID: u32 = 270;

/// Reserved range for future ADE tokens.
pub const RESERVED_START: u32 = 271;
pub const RESERVED_END: u32 = 2999;

/// BPE merge tokens start here.
pub const BPE_MERGE_START: u32 = 3000;
pub const BPE_MERGE_END: u32 = 63_999;
pub const BPE_MERGE_COUNT: usize = (BPE_MERGE_END - BPE_MERGE_START + 1) as usize;

/// Special token strings, ordered by ID (256..).
pub const SPECIAL_TOKENS: &[&str] = &[
    "<bos>", "<eos>", "<pad>", "<unk>",
    "<user>", "<assistant>", "<system>",
    "<tool_call>", "<tool_result>",
    "<think>", "</think>",
    "<memory>", "</memory>",
    "<code>", "</code>",
];

/// Returns the byte-fallback token bytes for a given ID (0-255).
pub fn byte_token(id: u32) -> Vec<u8> {
    debug_assert!(id <= 255);
    vec![id as u8]
}

/// Build the full base vocabulary: 256 byte tokens + 15 special tokens + reserved slots.
/// Returns tokens as byte sequences, ordered by ID.
pub fn build_base_vocab() -> Vec<Vec<u8>> {
    let mut tokens: Vec<Vec<u8>> = Vec::with_capacity(BPE_MERGE_START as usize);

    // 0-255: raw byte fallback
    for b in 0u8..=255 {
        tokens.push(vec![b]);
    }

    // 256-270: special tokens (stored as UTF-8 bytes)
    for &s in SPECIAL_TOKENS {
        tokens.push(s.as_bytes().to_vec());
    }

    // 271-2999: reserved (placeholder bytes)
    for i in RESERVED_START..=RESERVED_END {
        tokens.push(format!("<reserved_{i}>").into_bytes());
    }

    tokens
}

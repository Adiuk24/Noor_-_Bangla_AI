// ---- Encoder tests (always compiled) ----

#[test]
fn test_encoder_byte_fallback() {
    let encoder = borno::encoder::BornoEncoder::from_byte_fallback();
    let text = "hello";
    let ids = encoder.encode(text);
    assert_eq!(ids, vec![104, 101, 108, 108, 111]);
    let decoded = encoder.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_encoder_special_tokens() {
    let encoder = borno::encoder::BornoEncoder::from_byte_fallback();
    assert_eq!(encoder.bos_id(), 256);
    assert_eq!(encoder.eos_id(), 257);
    assert_eq!(encoder.vocab_size(), 64_000);
}

// ---- Borno API tests (always compiled) ----

#[test]
fn test_borno_api_encode_decode_ascii() {
    let borno = borno::Borno::from_byte_fallback();
    let text = "Hello, world!";
    let ids = borno.encode(text);
    let decoded = borno.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_borno_api_encode_decode_bangla() {
    let borno = borno::Borno::from_byte_fallback();
    let text = "নূর";
    let ids = borno.encode(text);
    assert!(!ids.is_empty());
    let decoded = borno.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_borno_api_encode_decode_mixed() {
    let borno = borno::Borno::from_byte_fallback();
    let text = "Hello নূর world";
    let ids = borno.encode(text);
    let decoded = borno.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_borno_api_special_tokens() {
    let borno = borno::Borno::from_byte_fallback();
    let text = "<bos>Hello<eos>";
    let ids = borno.encode(text);
    assert_eq!(ids[0], borno::vocab::BOS_ID);
    assert_eq!(*ids.last().unwrap(), borno::vocab::EOS_ID);
}

#[test]
fn test_borno_api_empty() {
    let borno = borno::Borno::from_byte_fallback();
    let empty: Vec<u32> = vec![];
    assert_eq!(borno.encode(""), empty);
    assert_eq!(borno.decode(&[]), "");
}

// ---- Trainer test (feature-gated) ----

#[cfg(feature = "train")]
#[test]
fn test_train_tiny_bpe() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("borno_test_train");
    std::fs::create_dir_all(&dir).unwrap();
    let corpus_path = dir.join("tiny_corpus.txt");
    let mut f = std::fs::File::create(&corpus_path).unwrap();
    for _ in 0..100 {
        writeln!(f, "hello world this is a test of the tokenizer").unwrap();
        writeln!(f, "আমি বাংলায় কথা বলি এটি একটি পরীক্ষা").unwrap();
        writeln!(f, "def foo(x): return x + 1").unwrap();
    }
    drop(f);
    let output_dir = dir.join("output");
    std::fs::create_dir_all(&output_dir).unwrap();
    let result = borno::trainer::train_bpe(
        &[corpus_path.to_str().unwrap().to_string()],
        &output_dir,
        500,
    );
    assert!(result.is_ok(), "Training failed: {:?}", result.err());
    assert!(output_dir.join("borno-vocab.json").exists());
    assert!(output_dir.join("borno-merges.txt").exists());
    std::fs::remove_dir_all(&dir).ok();
}

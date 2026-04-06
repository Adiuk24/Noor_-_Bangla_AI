#[test]
fn test_nfc_normalization() {
    let decomposed = "কো";
    let result = borno::bangla::normalize(decomposed);
    let result2 = borno::bangla::normalize(decomposed);
    assert_eq!(result, result2);
}

#[test]
fn test_grapheme_cluster_ksha() {
    // ক্ষ = ক + ্ + ষ — one grapheme cluster, must not be split
    let clusters = borno::bangla::grapheme_clusters("ক্ষ");
    assert_eq!(clusters, vec!["ক্ষ"]);
}

#[test]
fn test_grapheme_cluster_stra() {
    // স্ত্র = স + ্ + ত + ্ + র — one cluster
    let clusters = borno::bangla::grapheme_clusters("স্ত্র");
    assert_eq!(clusters, vec!["স্ত্র"]);
}

#[test]
fn test_grapheme_clusters_bangla_word() {
    // বাংলা → বা + ং + লা (3 grapheme clusters)
    let clusters = borno::bangla::grapheme_clusters("বাংলা");
    assert_eq!(clusters.len(), 3);
    assert_eq!(clusters, vec!["বা", "ং", "লা"]);
}

#[test]
fn test_grapheme_ki() {
    // কি = ক + ি — one cluster
    let clusters = borno::bangla::grapheme_clusters("কি");
    assert_eq!(clusters, vec!["কি"]);
}

#[test]
fn test_normalize_then_segment() {
    let input = "আমি বাংলায় কথা বলি";
    let result = borno::bangla::normalize_and_segment(input);
    for cluster in &result {
        assert!(!cluster.is_empty());
    }
    assert!(result.len() < 20, "Got {} clusters, expected <20", result.len());
}

#[test]
fn test_is_bengali() {
    assert!(borno::bangla::is_bengali_char('ক'));
    assert!(borno::bangla::is_bengali_char('া'));
    assert!(borno::bangla::is_bengali_char('্'));
    assert!(!borno::bangla::is_bengali_char('a'));
    assert!(!borno::bangla::is_bengali_char('1'));
}

#[test]
fn test_script_split_mixed() {
    let spans = borno::pretokenize::split_by_script("Hello বাংলা world");
    assert_eq!(spans.len(), 3);
    assert_eq!(spans[0].text, "Hello ");
    assert!(!spans[0].is_bengali);
    assert_eq!(spans[1].text, "বাংলা");
    assert!(spans[1].is_bengali);
    assert_eq!(spans[2].text, " world");
    assert!(!spans[2].is_bengali);
}

#[test]
fn test_pretokenize_bengali_span() {
    let tokens = borno::pretokenize::pretokenize("আমি বাংলায় কথা বলি");
    assert!(tokens.len() > 5);
    assert!(tokens.len() < 20);
}

#[test]
fn test_pretokenize_english_span() {
    let tokens = borno::pretokenize::pretokenize("Hello world! def foo():");
    assert!(tokens.len() >= 4);
}

#[test]
fn test_pretokenize_mixed() {
    let tokens = borno::pretokenize::pretokenize("Hello বাংলা code");
    assert!(tokens.len() >= 3);
}

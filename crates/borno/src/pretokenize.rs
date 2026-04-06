use crate::bangla;

#[derive(Debug, Clone, PartialEq)]
pub struct ScriptSpan {
    pub text: String,
    pub is_bengali: bool,
}

pub fn split_by_script(input: &str) -> Vec<ScriptSpan> {
    if input.is_empty() {
        return vec![];
    }

    let mut spans: Vec<ScriptSpan> = Vec::new();
    // `current` holds chars accumulated for the current script run.
    let mut current = String::new();
    let mut current_is_bengali: Option<bool> = None;
    // `neutral_buf` collects whitespace/punctuation between script runs.
    // When we see the next non-neutral char we decide where neutrals go:
    //   - If we're about to enter a Bengali run, the neutrals attach to the
    //     PRECEDING Latin span (flush current+neutrals together, then start Bengali).
    //   - If we're about to enter a Latin run, the neutrals attach to the
    //     FOLLOWING Latin span (flush current without neutrals, prepend neutrals
    //     to the new Latin span).
    // In both cases: neutrals go with the Latin (non-Bengali) side.
    let mut neutral_buf = String::new();

    for c in input.chars() {
        let is_b = bangla::is_bengali_char(c);
        let is_neutral = c.is_whitespace() || c.is_ascii_punctuation();

        if is_neutral {
            neutral_buf.push(c);
            continue;
        }

        match current_is_bengali {
            None => {
                // First non-neutral char. Neutrals go with whatever starts first.
                current_is_bengali = Some(is_b);
                current.push_str(&neutral_buf);
                neutral_buf.clear();
                current.push(c);
            }
            Some(was_bengali) if was_bengali == is_b => {
                // Same script: neutrals stay in current span.
                current.push_str(&neutral_buf);
                neutral_buf.clear();
                current.push(c);
            }
            Some(was_bengali) => {
                // Script change.
                if is_b {
                    // Switching Latin -> Bengali:
                    // Neutrals attach to the PRECEDING Latin span.
                    current.push_str(&neutral_buf);
                    neutral_buf.clear();
                    spans.push(ScriptSpan {
                        text: current.clone(),
                        is_bengali: was_bengali,
                    });
                    current.clear();
                    current_is_bengali = Some(true);
                    current.push(c);
                } else {
                    // Switching Bengali -> Latin:
                    // Neutrals attach to the FOLLOWING Latin span.
                    spans.push(ScriptSpan {
                        text: current.clone(),
                        is_bengali: was_bengali,
                    });
                    current.clear();
                    current_is_bengali = Some(false);
                    // Prepend buffered neutrals to the new Latin span.
                    current.push_str(&neutral_buf);
                    neutral_buf.clear();
                    current.push(c);
                }
            }
        }
    }

    // Flush remaining neutrals into current span.
    if !neutral_buf.is_empty() {
        current.push_str(&neutral_buf);
    }

    if !current.is_empty() {
        spans.push(ScriptSpan {
            text: current,
            is_bengali: current_is_bengali.unwrap_or(false),
        });
    }

    spans
}

fn split_latin(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if c.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            current.push(c);
        } else if c.is_ascii_punctuation() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            tokens.push(c.to_string());
        } else {
            current.push(c);
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

pub fn pretokenize(input: &str) -> Vec<String> {
    let normalized = bangla::normalize(input);
    let spans = split_by_script(&normalized);
    let mut result = Vec::new();

    for span in spans {
        if span.is_bengali {
            let clusters = bangla::grapheme_clusters(&span.text);
            result.extend(clusters.into_iter().map(String::from));
        } else {
            result.extend(split_latin(&span.text));
        }
    }

    result
}

# Rust Style Guide

Based on the official Rust Style Guide: https://doc.rust-lang.org/style-guide/

## Core Principles
- **Idiomatic Rust:** Always prefer idiomatic solutions (e.g., using 'Iterator' methods over manual loops where appropriate).
- **Safety First:** Minimize 'unsafe' blocks and document their necessity thoroughly.
- **Performance:** Leverage zero-cost abstractions and move semantics.

## Formatting
- **rustfmt:** Use 'rustfmt' with default settings for all formatting.
- **Naming:** Follow 'snake_case' for functions/variables and 'PascalCase' for types/traits.

## Patterns
- **Error Handling:** Use 'Result' and 'Option' types; avoid 'unwrap()' or 'expect()' in production code.
- **Documentation:** Use '///' for doc comments on all public items.

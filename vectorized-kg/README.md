# vectorized-kg

A vectorized knowledge graph implementation in Rust using ndarray for vector operations.

## Features

- Text and keyword nodes with embeddings
- U matrix for text-keyword relationships
- Similarity search (cosine distance)
- Disk persistence via JSON serialization
- Deterministic embeddings using seeded RNG

## Installation

```bash
git clone https://github.com/manupatet/clawz.git
cd clawz/vectorized-kg
```

Add this to your `Cargo.toml`:

```toml
[dependencies]
vectorized-kg = { path = "path/to/clawz/vectorized-kg" }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
ndarray = "0.15"
tracing = "0.1"
rand_chacha = "0.3"
rand_core = "0.6"
```

## Usage

```rust
use vectorized_kg::{NumpyGraphStore, Document, SourceInfo, GraphConfig};

let config = GraphConfig::default();
let documents = vec![
    Document {
        text: "Hello world".to_string(),
        source: SourceInfo {
            filename: "test.txt".to_string(),
            page_num: Some(1),
            file_type: "txt".to_string(),
            chunk_idx: Some(0),
        },
    },
];

let mut store = NumpyGraphStore::new();
store.build_kg(&documents, &config);

// Search similar texts
let query_vec = vec![0.0; config.embedding_dim];
let results = store.search_similar_texts(&query_vec, 5);

// Save to disk
store.save("graph.json").unwrap();

// Load from disk
let loaded = NumpyGraphStore::load("graph.json").unwrap();
```

## Running Tests

```bash
cd vectorized-kg
cargo test
```
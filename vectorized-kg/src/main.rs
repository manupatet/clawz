use ndarray::Array2;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hash::{Hash, Hasher};
use serde::{Deserialize, Serialize};

/// Represents a text node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextNode {
    pub id: usize,
    pub text: String,
    pub source: SourceInfo,
    pub embedding: Vec<f32>,
    pub token_count: usize,
}

/// Source information with provenance
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SourceInfo {
    pub filename: String,
    pub page_num: Option<u32>,
    pub file_type: String,
    pub chunk_idx: Option<usize>,
}

/// Represents a keyword node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordNode {
    pub id: usize,
    pub text: String,
    pub embedding: Vec<f32>,
}

/// In-memory graph store using ndarray for vector operations
#[derive(Debug, Clone)]
pub struct NumpyGraphStore {
    texts: Vec<TextNode>,
    keywords: Vec<KeywordNode>,
    u_mat: Option<Array2<f32>>,
    pred_mat: Option<Array2<u8>>,
}

impl NumpyGraphStore {
    pub fn new() -> Self {
        Self {
            texts: Vec::new(),
            keywords: Vec::new(),
            u_mat: None,
            pred_mat: None,
        }
    }

    /// Build knowledge graph from documents
    pub fn build_kg(&mut self, documents: &[Document], config: &GraphConfig) {
        tracing::info!("Building knowledge graph from {} documents...", documents.len());

        let texts: Vec<String> = documents.iter().map(|d| d.text.clone()).collect();
        let sources: Vec<SourceInfo> = documents.iter().map(|d| d.source.clone()).collect();

        tracing::info!("Generating embeddings...");
        let vectors: Vec<Vec<f32>> = self.mock_embeddings(&texts, config.embedding_dim);
        let token_counts: Vec<usize> = texts.iter().map(|t| t.split_whitespace().count()).collect();

        tracing::info!("Removing duplicate texts...");
        let (texts, sources, vectors, token_counts) =
            self.remove_duplicates(texts, sources, vectors, token_counts);

        tracing::info!("After deduplication: {} texts", texts.len());

        self.texts = texts
            .into_iter()
            .enumerate()
            .map(|(id, text)| TextNode {
                id,
                text,
                source: sources[id].clone(),
                embedding: vectors[id].clone(),
                token_count: token_counts[id],
            })
            .collect();

        tracing::info!("Extracting keywords...");
        let keywords = self.extract_keywords(&self.texts);
        tracing::info!("Extracted {} unique keywords", keywords.len());

        let keyvectors: Vec<Vec<f32>> = self.mock_embeddings(&keywords, config.embedding_dim);

        self.keywords = keywords
            .into_iter()
            .enumerate()
            .map(|(id, text)| KeywordNode {
                id,
                text,
                embedding: keyvectors[id].clone(),
            })
            .collect();

        tracing::info!("Building keyword relationships...");
        self.build_keyword_relationships();
    }

    fn mock_embeddings(&self, texts: &[String], dim: usize) -> Vec<Vec<f32>> {
        // Deterministic embeddings based on text hash
        texts
            .iter()
            .map(|text| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                text.hash(&mut hasher);
                let seed = hasher.finish() as u64;
                let mut local_rng = ChaCha8Rng::seed_from_u64(seed);
                let mut vec = vec![0.0f32; dim];
                for v in vec.iter_mut() {
                    *v = (local_rng.next_u32() as f32) / (u32::MAX as f32) * 2.0 - 1.0;
                }
                // Normalize
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    vec.iter_mut().for_each(|x| *x /= norm);
                }
                vec
            })
            .collect()
    }

    fn remove_duplicates(
        &self,
        texts: Vec<String>,
        sources: Vec<SourceInfo>,
        vectors: Vec<Vec<f32>>,
        token_counts: Vec<usize>,
    ) -> (Vec<String>, Vec<SourceInfo>, Vec<Vec<f32>>, Vec<usize>) {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        for (i, text) in texts.into_iter().enumerate() {
            if !seen.contains(&text) {
                seen.insert(text.clone());
                result.push((text, sources[i].clone(), vectors[i].clone(), token_counts[i]));
            }
        }

        let texts: Vec<String> = result.iter().map(|(t, _, _, _)| t.clone()).collect();
        let sources: Vec<SourceInfo> = result.iter().map(|(_, s, _, _)| s.clone()).collect();
        let vectors: Vec<Vec<f32>> = result.iter().map(|(_, _, v, _)| v.clone()).collect();
        let token_counts: Vec<usize> = result.iter().map(|(_, _, _, c)| *c).collect();

        (texts, sources, vectors, token_counts)
    }

    fn extract_keywords(&self, texts: &[TextNode]) -> Vec<String> {
        // Simple placeholder - extract common words
        let mut keywords = std::collections::HashSet::new();

        for text_node in texts {
            for word in text_node.text.split_whitespace() {
                if word.len() > 3 {
                    keywords.insert(word.to_lowercase());
                }
            }
        }

        keywords.into_iter().collect()
    }

    fn build_keyword_relationships(&mut self) {
        let n_texts = self.texts.len();
        let n_keywords = self.keywords.len();

        if n_texts == 0 || n_keywords == 0 {
            return;
        }

        // Create U matrix for demonstration
        let mut u_mat = Array2::zeros((n_texts, n_keywords));
        for i in 0..n_texts {
            for j in 0..n_keywords {
                u_mat[[i, j]] = (i + j) as f32 / (n_texts + n_keywords) as f32;
            }
        }

        self.u_mat = Some(u_mat);
    }

    pub fn search_similar_texts(&self, query_vec: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.texts.is_empty() {
            return Vec::new();
        }

        let mut distances: Vec<(usize, f32)> = self
            .texts
            .iter()
            .enumerate()
            .map(|(i, text)| (i, cosine_distance(query_vec, &text.embedding)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let k = std::cmp::min(k, self.texts.len());
        distances.truncate(k);
        distances
    }

    pub fn search_similar_keywords(&self, query_vec: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.keywords.is_empty() {
            return Vec::new();
        }

        let mut distances: Vec<(usize, f32)> = self
            .keywords
            .iter()
            .enumerate()
            .map(|(i, kw)| (i, cosine_distance(query_vec, &kw.embedding)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let k = std::cmp::min(k, self.keywords.len());
        distances.truncate(k);
        distances
    }

    pub fn get_keyword_related_texts(&self, keyword_idx: usize, k: usize) -> Vec<usize> {
        if let Some(u_mat) = &self.u_mat {
            if keyword_idx >= u_mat.ncols() {
                return Vec::new();
            }

            let mut scores: Vec<(usize, f32)> = (0..u_mat.nrows())
                .map(|i| (i, u_mat[[i, keyword_idx]]))
                .collect();

            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let k = std::cmp::min(k, u_mat.nrows() as usize);
            scores.truncate(k);
            scores.into_iter().map(|(i, _)| i).collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_adjacent_keywords(&self, keyword_idx: usize, k: usize) -> Vec<usize> {
        if self.keywords.is_empty() {
            return Vec::new();
        }

        let mut adjacent: Vec<usize> = (0..self.keywords.len())
            .filter(|&i| i != keyword_idx)
            .collect();

        let k = std::cmp::min(k, adjacent.len());
        adjacent.truncate(k);
        adjacent
    }

    pub fn get_texts(&self) -> &[TextNode] {
        &self.texts
    }

    pub fn get_keywords(&self) -> &[KeywordNode] {
        &self.keywords
    }

    pub fn get_sources(&self) -> Vec<&SourceInfo> {
        self.texts.iter().map(|t| &t.source).collect::<Vec<_>>()
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serializable = GraphStoreSnapshot {
            texts: self.texts.clone(),
            keywords: self.keywords.clone(),
        };
        let data = serde_json::to_string_pretty(&serializable)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let snapshot: GraphStoreSnapshot = serde_json::from_str(&data)?;
        Ok(Self {
            texts: snapshot.texts,
            keywords: snapshot.keywords,
            u_mat: None,
            pred_mat: None,
        })
    }
}

/// Compute cosine distance between two vectors
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 1.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - dot / (norm_a * norm_b)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub text: String,
    pub source: SourceInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    pub embedding_dim: usize,
    pub k_neighbors: usize,
    pub trust_num: usize,
    pub negative_multiplier: usize,
    pub connect_threshold: f32,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 768,
            k_neighbors: 30,
            trust_num: 5,
            negative_multiplier: 7,
            connect_threshold: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphStoreSnapshot {
    texts: Vec<TextNode>,
    keywords: Vec<KeywordNode>,
}

fn main() {
    println!("vectorized-kg: Knowledge graph implementation for Pingoo");
    
    let config = GraphConfig::default();
    
    let documents = vec![
        Document {
            text: "Hello world from Pingoo".to_string(),
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
    
    println!("Built graph with {} texts and {} keywords", 
             store.get_texts().len(), 
             store.get_keywords().len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_graph() {
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
        assert!(store.get_texts().len() > 0);
    }

    #[test]
    fn test_save_load() {
        let config = GraphConfig::default();
        let documents = vec![Document {
            text: "Test document".to_string(),
            source: SourceInfo {
                filename: "test.txt".to_string(),
                page_num: Some(1),
                file_type: "txt".to_string(),
                chunk_idx: Some(0),
            },
        }];
        let mut store = NumpyGraphStore::new();
        store.build_kg(&documents, &config);
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("graph.json");
        store.save(path.to_str().unwrap()).unwrap();
        let loaded = NumpyGraphStore::load(path.to_str().unwrap()).unwrap();
        assert_eq!(store.get_texts().len(), loaded.get_texts().len());
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_bounds() {
        let config = GraphConfig::default();
        let documents = vec![
            Document {
                text: "Doc 1".to_string(),
                source: SourceInfo {
                    filename: "doc1.txt".to_string(),
                    page_num: Some(1),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(0),
                },
            },
            Document {
                text: "Doc 2".to_string(),
                source: SourceInfo {
                    filename: "doc2.txt".to_string(),
                    page_num: Some(2),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(1),
                },
            },
        ];
        let mut store = NumpyGraphStore::new();
        store.build_kg(&documents, &config);

        // Test k larger than dataset
        let query_vec = vec![0.0; config.embedding_dim];
        let results = store.search_similar_texts(&query_vec, 100);
        assert_eq!(results.len(), store.get_texts().len());
    }

    #[test]
    fn test_empty_store_search() {
        let store = NumpyGraphStore::new();
        let query_vec = vec![0.0; 128];
        
        let text_results = store.search_similar_texts(&query_vec, 5);
        assert!(text_results.is_empty());
    }

    #[test]
    fn test_zero_k_search() {
        let config = GraphConfig::default();
        let documents = vec![
            Document {
                text: "Doc 1".to_string(),
                source: SourceInfo {
                    filename: "doc1.txt".to_string(),
                    page_num: Some(1),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(0),
                },
            },
        ];
        let mut store = NumpyGraphStore::new();
        store.build_kg(&documents, &config);

        let query_vec = vec![0.0; config.embedding_dim];
        let results = store.search_similar_texts(&query_vec, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_monotonic_distances() {
        let config = GraphConfig::default();
        let documents = vec![
            Document {
                text: "Doc 1".to_string(),
                source: SourceInfo {
                    filename: "doc1.txt".to_string(),
                    page_num: Some(1),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(0),
                },
            },
            Document {
                text: "Doc 2".to_string(),
                source: SourceInfo {
                    filename: "doc2.txt".to_string(),
                    page_num: Some(2),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(1),
                },
            },
            Document {
                text: "Doc 3".to_string(),
                source: SourceInfo {
                    filename: "doc3.txt".to_string(),
                    page_num: Some(3),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(2),
                },
            },
        ];
        let mut store = NumpyGraphStore::new();
        store.build_kg(&documents, &config);

        let query_vec = vec![0.0; config.embedding_dim];
        let results = store.search_similar_texts(&query_vec, 5);

        // Check distances are non-decreasing
        for i in 0..results.len().saturating_sub(1) {
            assert!(results[i].1 <= results[i + 1].1 + 1e-6);
        }
    }

    #[test]
    fn test_finite_distances() {
        let config = GraphConfig::default();
        let documents = vec![
            Document {
                text: "Doc 1".to_string(),
                source: SourceInfo {
                    filename: "doc1.txt".to_string(),
                    page_num: Some(1),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(0),
                },
            },
        ];
        let mut store = NumpyGraphStore::new();
        store.build_kg(&documents, &config);

        let query_vec = vec![0.0; config.embedding_dim];
        let results = store.search_similar_texts(&query_vec, 5);

        for (_, dist) in results {
            assert!(dist.is_finite());
        }
    }

    #[test]
    fn test_duplicate_removal() {
        let config = GraphConfig::default();
        let documents = vec![
            Document {
                text: "Unique text".to_string(),
                source: SourceInfo {
                    filename: "doc1.txt".to_string(),
                    page_num: Some(1),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(0),
                },
            },
            Document {
                text: "Unique text".to_string(), // Duplicate
                source: SourceInfo {
                    filename: "doc2.txt".to_string(),
                    page_num: Some(2),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(1),
                },
            },
        ];
        let mut store = NumpyGraphStore::new();
        store.build_kg(&documents, &config);

        // Should have removed duplicate
        assert_eq!(store.get_texts().len(), 1);
    }

    #[test]
    fn test_provenance_preserved() {
        let config = GraphConfig::default();
        let documents = vec![
            Document {
                text: "Test document".to_string(),
                source: SourceInfo {
                    filename: "test.pdf".to_string(),
                    page_num: Some(5),
                    file_type: "pdf".to_string(),
                    chunk_idx: Some(2),
                },
            },
        ];
        let mut store = NumpyGraphStore::new();
        store.build_kg(&documents, &config);

        let sources = store.get_sources();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].filename, "test.pdf");
        assert_eq!(sources[0].page_num, Some(5));
        assert_eq!(sources[0].file_type, "pdf");
        assert_eq!(sources[0].chunk_idx, Some(2));
    }

    #[test]
    fn test_keyword_related_texts_bounds() {
        let config = GraphConfig::default();
        let documents = vec![
            Document {
                text: "Doc 1 with some words".to_string(),
                source: SourceInfo {
                    filename: "doc1.txt".to_string(),
                    page_num: Some(1),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(0),
                },
            },
            Document {
                text: "Doc 2 with different words".to_string(),
                source: SourceInfo {
                    filename: "doc2.txt".to_string(),
                    page_num: Some(2),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(1),
                },
            },
        ];
        let mut store = NumpyGraphStore::new();
        store.build_kg(&documents, &config);

        // Test with k larger than available texts
        if !store.get_keywords().is_empty() {
            let keyword_idx = 0;
            let results = store.get_keyword_related_texts(keyword_idx, 100);
            assert!(results.len() <= store.get_texts().len());
        }
    }

    #[test]
    fn test_adjacent_keywords_bounds() {
        let config = GraphConfig::default();
        let documents = vec![
            Document {
                text: "Doc 1".to_string(),
                source: SourceInfo {
                    filename: "doc1.txt".to_string(),
                    page_num: Some(1),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(0),
                },
            },
        ];
        let mut store = NumpyGraphStore::new();
        store.build_kg(&documents, &config);

        if !store.get_keywords().is_empty() {
            let keyword_idx = 0;
            let adjacent = store.get_adjacent_keywords(keyword_idx, 100);
            // Should be bounded by available keywords (excluding self)
            assert!(adjacent.len() <= store.get_keywords().len().saturating_sub(1));
        }
    }

    #[test]
    fn test_identical_queries_deterministic() {
        let config = GraphConfig::default();
        let documents = vec![
            Document {
                text: "Doc 1".to_string(),
                source: SourceInfo {
                    filename: "doc1.txt".to_string(),
                    page_num: Some(1),
                    file_type: "txt".to_string(),
                    chunk_idx: Some(0),
                },
            },
        ];
        let mut store = NumpyGraphStore::new();
        store.build_kg(&documents, &config);

        let query_vec = vec![0.0; config.embedding_dim];

        // Run query multiple times
        let results1 = store.search_similar_texts(&query_vec, 5);
        let results2 = store.search_similar_texts(&query_vec, 5);

        assert_eq!(results1.len(), results2.len());
        for i in 0..results1.len() {
            assert!((results1[i].1 - results2[i].1).abs() < 1e-6);
        }
    }
}
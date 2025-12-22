//! Inverted index for sparse vector similarity search.
//!
//! This module provides an inverted index implementation for SPLADE-style sparse vectors,
//! enabling efficient top-k retrieval using algorithms like WAND and DAAT.
//!
//! # Storage Tables
//!
//! - `inverted_postings`: Posting lists mapping token_id → [(point_id, weight), ...]
//! - `inverted_meta`: Index metadata (doc count, statistics)
//! - `inverted_point_tokens`: Reverse mapping point_id → [token_ids] for deletion
//!
//! # Search Algorithms
//!
//! - **DAAT (Document-at-a-time)**: Exact scoring by traversing all posting lists
//! - **WAND (Weak AND)**: Top-k retrieval with early termination
//!
//! # Scoring Functions
//!
//! - **Dot product**: Standard for SPLADE vectors
//! - **BM25-style**: Optional term frequency normalization
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_vector::store::InvertedIndex;
//! use manifoldb_core::PointId;
//!
//! let index = InvertedIndex::new(engine);
//!
//! // Index a sparse vector
//! let vector = vec![(100, 0.5), (200, 0.3), (300, 0.2)];
//! index.insert("documents", "keywords", PointId::new(1), &vector)?;
//!
//! // Search for similar vectors
//! let query = vec![(100, 1.0), (200, 0.8)];
//! let results = index.search_wand("documents", "keywords", &query, 10)?;
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Bound;

use manifoldb_core::PointId;
use manifoldb_storage::{Cursor, StorageEngine, Transaction};

use crate::encoding::{
    encode_inverted_meta_collection_prefix, encode_inverted_meta_key,
    encode_point_tokens_collection_prefix, encode_point_tokens_key, encode_point_tokens_prefix,
    encode_posting_collection_prefix, encode_posting_key, encode_posting_prefix,
};
use crate::error::VectorError;

/// Table name for posting lists.
const TABLE_POSTINGS: &str = "inverted_postings";

/// Table name for index metadata.
const TABLE_META: &str = "inverted_meta";

/// Table name for point-to-tokens reverse mapping.
const TABLE_POINT_TOKENS: &str = "inverted_point_tokens";

/// A posting list entry: (point_id, weight).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostingEntry {
    /// The point ID.
    pub point_id: PointId,
    /// The weight (term frequency or TF-IDF weight).
    pub weight: f32,
}

impl PostingEntry {
    /// Create a new posting entry.
    #[must_use]
    pub const fn new(point_id: PointId, weight: f32) -> Self {
        Self { point_id, weight }
    }
}

/// A posting list for a single token.
#[derive(Debug, Clone, Default)]
pub struct PostingList {
    /// Entries sorted by point_id for efficient merging.
    entries: Vec<PostingEntry>,
    /// Maximum weight in this posting list (for WAND upper bound).
    max_weight: f32,
}

impl PostingList {
    /// Create an empty posting list.
    #[must_use]
    pub const fn new() -> Self {
        Self { entries: Vec::new(), max_weight: 0.0 }
    }

    /// Create a posting list from entries.
    #[must_use]
    pub fn from_entries(mut entries: Vec<PostingEntry>) -> Self {
        entries.sort_by_key(|e| e.point_id.as_u64());
        let max_weight = entries.iter().map(|e| e.weight).fold(0.0f32, f32::max);
        Self { entries, max_weight }
    }

    /// Get the entries.
    #[must_use]
    pub fn entries(&self) -> &[PostingEntry] {
        &self.entries
    }

    /// Get the maximum weight.
    #[must_use]
    pub fn max_weight(&self) -> f32 {
        self.max_weight
    }

    /// Get the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add an entry, maintaining sort order.
    pub fn add(&mut self, entry: PostingEntry) {
        match self.entries.binary_search_by_key(&entry.point_id.as_u64(), |e| e.point_id.as_u64()) {
            Ok(idx) => {
                // Update existing entry
                self.entries[idx] = entry;
            }
            Err(idx) => {
                // Insert new entry
                self.entries.insert(idx, entry);
            }
        }
        self.max_weight = self.max_weight.max(entry.weight);
    }

    /// Remove an entry by point_id.
    pub fn remove(&mut self, point_id: PointId) -> bool {
        match self.entries.binary_search_by_key(&point_id.as_u64(), |e| e.point_id.as_u64()) {
            Ok(idx) => {
                let removed = self.entries.remove(idx);
                // Recalculate max_weight if we removed the max
                if (removed.weight - self.max_weight).abs() < f32::EPSILON {
                    self.max_weight = self.entries.iter().map(|e| e.weight).fold(0.0f32, f32::max);
                }
                true
            }
            Err(_) => false,
        }
    }

    /// Serialize to bytes.
    ///
    /// Format: [count: u32][max_weight: f32][(point_id: u64, weight: f32), ...]
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + self.entries.len() * 12);
        bytes.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.max_weight.to_le_bytes());
        for entry in &self.entries {
            bytes.extend_from_slice(&entry.point_id.as_u64().to_le_bytes());
            bytes.extend_from_slice(&entry.weight.to_le_bytes());
        }
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        if bytes.len() < 8 {
            return Err(VectorError::Encoding("posting list too short".to_string()));
        }

        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let max_weight = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        let expected_len = 8 + count * 12;
        if bytes.len() != expected_len {
            return Err(VectorError::Encoding(format!(
                "posting list length mismatch: expected {}, got {}",
                expected_len,
                bytes.len()
            )));
        }

        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let offset = 8 + i * 12;
            let point_id = u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]);
            let weight = f32::from_le_bytes([
                bytes[offset + 8],
                bytes[offset + 9],
                bytes[offset + 10],
                bytes[offset + 11],
            ]);
            entries.push(PostingEntry::new(PointId::new(point_id), weight));
        }

        Ok(Self { entries, max_weight })
    }
}

/// Inverted index metadata for a vector.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InvertedIndexMeta {
    /// Number of documents indexed.
    pub doc_count: u64,
    /// Total number of tokens across all documents.
    pub total_tokens: u64,
    /// Average document length (number of non-zero tokens).
    pub avg_doc_length: f32,
}

impl InvertedIndexMeta {
    /// Create new metadata.
    #[must_use]
    pub const fn new() -> Self {
        Self { doc_count: 0, total_tokens: 0, avg_doc_length: 0.0 }
    }

    /// Update statistics after adding a document.
    pub fn add_document(&mut self, token_count: usize) {
        self.doc_count += 1;
        self.total_tokens += token_count as u64;
        self.avg_doc_length = self.total_tokens as f32 / self.doc_count as f32;
    }

    /// Update statistics after removing a document.
    pub fn remove_document(&mut self, token_count: usize) {
        if self.doc_count > 0 {
            self.doc_count -= 1;
            self.total_tokens = self.total_tokens.saturating_sub(token_count as u64);
            if self.doc_count > 0 {
                self.avg_doc_length = self.total_tokens as f32 / self.doc_count as f32;
            } else {
                self.avg_doc_length = 0.0;
            }
        }
    }

    /// Serialize to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serde::encode_to_vec(self, bincode::config::standard()).unwrap_or_default()
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map(|(v, _)| v)
            .map_err(|e| VectorError::Encoding(format!("failed to deserialize index meta: {}", e)))
    }
}

impl Default for InvertedIndexMeta {
    fn default() -> Self {
        Self::new()
    }
}

/// Scoring function for sparse vector similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScoringFunction {
    /// Dot product: sum of query_weight * doc_weight for matching tokens.
    DotProduct,
    /// BM25-style scoring with length normalization.
    Bm25 {
        /// BM25 k1 parameter (default: 1.2).
        k1_times_10: u8,
        /// BM25 b parameter (default: 0.75).
        b_times_100: u8,
    },
}

impl Default for ScoringFunction {
    fn default() -> Self {
        Self::DotProduct
    }
}

impl ScoringFunction {
    /// Create BM25 scoring with default parameters.
    #[must_use]
    pub const fn bm25() -> Self {
        Self::Bm25 { k1_times_10: 12, b_times_100: 75 }
    }

    /// Create BM25 scoring with custom parameters.
    #[must_use]
    pub fn bm25_custom(k1: f32, b: f32) -> Self {
        Self::Bm25 {
            k1_times_10: (k1 * 10.0).clamp(0.0, 255.0) as u8,
            b_times_100: (b * 100.0).clamp(0.0, 255.0) as u8,
        }
    }
}

/// A search result with point ID and score.
#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    /// The point ID.
    pub point_id: PointId,
    /// The similarity score.
    pub score: f32,
}

impl SearchResult {
    /// Create a new search result.
    #[must_use]
    pub const fn new(point_id: PointId, score: f32) -> Self {
        Self { point_id, score }
    }
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.point_id == other.point_id
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (we want to pop smallest scores)
        other.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal)
    }
}

/// Inverted index for sparse vector similarity search.
pub struct InvertedIndex<E: StorageEngine> {
    engine: E,
}

impl<E: StorageEngine> InvertedIndex<E> {
    /// Create a new inverted index with the given storage engine.
    #[must_use]
    pub const fn new(engine: E) -> Self {
        Self { engine }
    }

    /// Get a reference to the storage engine.
    #[must_use]
    pub fn engine(&self) -> &E {
        &self.engine
    }

    // ========================================================================
    // Index operations
    // ========================================================================

    /// Insert a sparse vector into the index.
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `vector_name` - Vector name within the collection
    /// * `point_id` - Point ID
    /// * `vector` - Sparse vector as (token_id, weight) pairs (must be sorted)
    pub fn insert(
        &self,
        collection: &str,
        vector_name: &str,
        point_id: PointId,
        vector: &[(u32, f32)],
    ) -> Result<(), VectorError> {
        if vector.is_empty() {
            return Ok(());
        }

        let mut tx = self.engine.begin_write()?;

        // Load or create metadata
        let meta_key = encode_inverted_meta_key(collection, vector_name);
        let mut meta = tx
            .get(TABLE_META, &meta_key)?
            .map(|bytes| InvertedIndexMeta::from_bytes(&bytes))
            .transpose()?
            .unwrap_or_default();

        // Store token IDs for this point (for deletion)
        let token_ids: Vec<u32> = vector.iter().map(|(idx, _)| *idx).collect();
        let point_tokens_key = encode_point_tokens_key(collection, vector_name, point_id);
        tx.put(TABLE_POINT_TOKENS, &point_tokens_key, &encode_token_ids(&token_ids))?;

        // Add to posting lists
        for &(token_id, weight) in vector {
            let posting_key = encode_posting_key(collection, vector_name, token_id);

            // Load existing posting list or create new
            let mut posting_list = tx
                .get(TABLE_POSTINGS, &posting_key)?
                .map(|bytes| PostingList::from_bytes(&bytes))
                .transpose()?
                .unwrap_or_default();

            posting_list.add(PostingEntry::new(point_id, weight));
            tx.put(TABLE_POSTINGS, &posting_key, &posting_list.to_bytes())?;
        }

        // Update metadata
        meta.add_document(vector.len());
        tx.put(TABLE_META, &meta_key, &meta.to_bytes())?;

        tx.commit()?;
        Ok(())
    }

    /// Delete a sparse vector from the index.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the vector was deleted, `Ok(false)` if it wasn't indexed.
    pub fn delete(
        &self,
        collection: &str,
        vector_name: &str,
        point_id: PointId,
    ) -> Result<bool, VectorError> {
        let mut tx = self.engine.begin_write()?;

        // Get token IDs for this point
        let point_tokens_key = encode_point_tokens_key(collection, vector_name, point_id);
        let token_ids = match tx.get(TABLE_POINT_TOKENS, &point_tokens_key)? {
            Some(bytes) => decode_token_ids(&bytes)?,
            None => return Ok(false),
        };

        // Load metadata
        let meta_key = encode_inverted_meta_key(collection, vector_name);
        let mut meta = tx
            .get(TABLE_META, &meta_key)?
            .map(|bytes| InvertedIndexMeta::from_bytes(&bytes))
            .transpose()?
            .unwrap_or_default();

        // Remove from posting lists
        for token_id in &token_ids {
            let posting_key = encode_posting_key(collection, vector_name, *token_id);

            if let Some(bytes) = tx.get(TABLE_POSTINGS, &posting_key)? {
                let mut posting_list = PostingList::from_bytes(&bytes)?;
                posting_list.remove(point_id);

                if posting_list.is_empty() {
                    tx.delete(TABLE_POSTINGS, &posting_key)?;
                } else {
                    tx.put(TABLE_POSTINGS, &posting_key, &posting_list.to_bytes())?;
                }
            }
        }

        // Delete point tokens mapping
        tx.delete(TABLE_POINT_TOKENS, &point_tokens_key)?;

        // Update metadata
        meta.remove_document(token_ids.len());
        tx.put(TABLE_META, &meta_key, &meta.to_bytes())?;

        tx.commit()?;
        Ok(true)
    }

    /// Update a sparse vector in the index (delete + insert).
    pub fn update(
        &self,
        collection: &str,
        vector_name: &str,
        point_id: PointId,
        vector: &[(u32, f32)],
    ) -> Result<(), VectorError> {
        // Delete existing
        self.delete(collection, vector_name, point_id)?;
        // Insert new
        self.insert(collection, vector_name, point_id, vector)
    }

    /// Delete all index data for a collection.
    pub fn delete_collection(&self, collection: &str) -> Result<(), VectorError> {
        let mut tx = self.engine.begin_write()?;

        // Delete all posting lists
        delete_by_prefix(&mut tx, TABLE_POSTINGS, &encode_posting_collection_prefix(collection))?;

        // Delete all point tokens
        delete_by_prefix(
            &mut tx,
            TABLE_POINT_TOKENS,
            &encode_point_tokens_collection_prefix(collection),
        )?;

        // Delete all metadata
        delete_by_prefix(&mut tx, TABLE_META, &encode_inverted_meta_collection_prefix(collection))?;

        tx.commit()?;
        Ok(())
    }

    /// Delete all index data for a specific vector in a collection.
    pub fn delete_vector(&self, collection: &str, vector_name: &str) -> Result<(), VectorError> {
        let mut tx = self.engine.begin_write()?;

        // Delete all posting lists for this vector
        delete_by_prefix(&mut tx, TABLE_POSTINGS, &encode_posting_prefix(collection, vector_name))?;

        // Delete all point tokens for this vector
        delete_by_prefix(
            &mut tx,
            TABLE_POINT_TOKENS,
            &encode_point_tokens_prefix(collection, vector_name),
        )?;

        // Delete metadata
        let meta_key = encode_inverted_meta_key(collection, vector_name);
        tx.delete(TABLE_META, &meta_key)?;

        tx.commit()?;
        Ok(())
    }

    // ========================================================================
    // Query operations
    // ========================================================================

    /// Get index metadata.
    pub fn get_meta(
        &self,
        collection: &str,
        vector_name: &str,
    ) -> Result<InvertedIndexMeta, VectorError> {
        let tx = self.engine.begin_read()?;
        let meta_key = encode_inverted_meta_key(collection, vector_name);
        tx.get(TABLE_META, &meta_key)?
            .map(|bytes| InvertedIndexMeta::from_bytes(&bytes))
            .transpose()?
            .ok_or_else(|| {
                VectorError::SpaceNotFound(format!("index '{}/{}'", collection, vector_name))
            })
    }

    /// Get a posting list for a specific token.
    pub fn get_posting_list(
        &self,
        collection: &str,
        vector_name: &str,
        token_id: u32,
    ) -> Result<Option<PostingList>, VectorError> {
        let tx = self.engine.begin_read()?;
        let posting_key = encode_posting_key(collection, vector_name, token_id);
        tx.get(TABLE_POSTINGS, &posting_key)?
            .map(|bytes| PostingList::from_bytes(&bytes))
            .transpose()
    }

    // ========================================================================
    // Search algorithms
    // ========================================================================

    /// Search using DAAT (Document-at-a-time) algorithm.
    ///
    /// This is exact scoring that traverses all posting lists to compute
    /// the full similarity score for each candidate document.
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `vector_name` - Vector name
    /// * `query` - Query vector as (token_id, weight) pairs
    /// * `top_k` - Number of results to return
    /// * `scoring` - Scoring function to use
    pub fn search_daat(
        &self,
        collection: &str,
        vector_name: &str,
        query: &[(u32, f32)],
        top_k: usize,
        scoring: ScoringFunction,
    ) -> Result<Vec<SearchResult>, VectorError> {
        if query.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let tx = self.engine.begin_read()?;

        // Load metadata for BM25
        let meta = if matches!(scoring, ScoringFunction::Bm25 { .. }) {
            let meta_key = encode_inverted_meta_key(collection, vector_name);
            tx.get(TABLE_META, &meta_key)?
                .map(|bytes| InvertedIndexMeta::from_bytes(&bytes))
                .transpose()?
        } else {
            None
        };

        // Load posting lists for all query tokens
        let mut posting_lists: Vec<(u32, f32, PostingList)> = Vec::with_capacity(query.len());
        for &(token_id, query_weight) in query {
            let posting_key = encode_posting_key(collection, vector_name, token_id);
            if let Some(bytes) = tx.get(TABLE_POSTINGS, &posting_key)? {
                let posting_list = PostingList::from_bytes(&bytes)?;
                if !posting_list.is_empty() {
                    posting_lists.push((token_id, query_weight, posting_list));
                }
            }
        }

        if posting_lists.is_empty() {
            return Ok(Vec::new());
        }

        // Accumulate scores for all documents
        let mut scores: HashMap<u64, f32> = HashMap::new();

        for (token_id, query_weight, posting_list) in &posting_lists {
            for entry in posting_list.entries() {
                let doc_id = entry.point_id.as_u64();
                let term_score = match scoring {
                    ScoringFunction::DotProduct => query_weight * entry.weight,
                    ScoringFunction::Bm25 { k1_times_10, b_times_100 } => {
                        let k1 = k1_times_10 as f32 / 10.0;
                        let b = b_times_100 as f32 / 100.0;
                        compute_bm25_term_score(
                            *query_weight,
                            entry.weight,
                            meta.as_ref(),
                            *token_id,
                            posting_list.len(),
                            k1,
                            b,
                        )
                    }
                };
                *scores.entry(doc_id).or_insert(0.0) += term_score;
            }
        }

        // Get top-k results
        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .map(|(doc_id, score)| SearchResult::new(PointId::new(doc_id), score))
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(top_k);

        Ok(results)
    }

    /// Search using WAND (Weak AND) algorithm.
    ///
    /// This is an optimized top-k search that uses upper bound scores
    /// to skip documents that cannot make it into the result set.
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `vector_name` - Vector name
    /// * `query` - Query vector as (token_id, weight) pairs
    /// * `top_k` - Number of results to return
    pub fn search_wand(
        &self,
        collection: &str,
        vector_name: &str,
        query: &[(u32, f32)],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, VectorError> {
        if query.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let tx = self.engine.begin_read()?;

        // Load posting lists for all query tokens with their upper bounds
        let mut cursors: Vec<WandCursor> = Vec::with_capacity(query.len());
        for &(token_id, query_weight) in query {
            let posting_key = encode_posting_key(collection, vector_name, token_id);
            if let Some(bytes) = tx.get(TABLE_POSTINGS, &posting_key)? {
                let posting_list = PostingList::from_bytes(&bytes)?;
                if !posting_list.is_empty() {
                    let upper_bound = query_weight * posting_list.max_weight();
                    cursors.push(WandCursor::new(posting_list, query_weight, upper_bound));
                }
            }
        }

        if cursors.is_empty() {
            return Ok(Vec::new());
        }

        // WAND algorithm
        let mut heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(top_k + 1);
        let mut threshold = 0.0f32;

        loop {
            // Sort cursors by current document ID
            cursors.sort_by_key(|c| c.current_doc_id());

            // Skip exhausted cursors and find first valid one
            let first_valid = cursors.iter().position(|c| !c.exhausted());
            if first_valid.is_none() {
                break;
            }

            // Find pivot: smallest index where sum of upper bounds >= threshold
            let mut upper_sum = 0.0f32;
            let mut pivot_idx = None;

            for (i, cursor) in cursors.iter().enumerate() {
                if cursor.exhausted() {
                    continue;
                }
                upper_sum += cursor.upper_bound;
                if upper_sum >= threshold {
                    pivot_idx = Some(i);
                    break;
                }
            }

            let pivot_idx = match pivot_idx {
                Some(idx) => idx,
                None => break, // No more candidates can beat threshold
            };

            let pivot_doc_id = cursors[pivot_idx].current_doc_id();

            // Check if all non-exhausted cursors before pivot point to the same document
            let all_aligned = cursors[..pivot_idx]
                .iter()
                .filter(|c| !c.exhausted())
                .all(|c| c.current_doc_id() == pivot_doc_id);

            if all_aligned || pivot_idx == 0 {
                // Score this document - include all cursors pointing to this doc
                let mut score = 0.0f32;
                for cursor in &cursors {
                    if !cursor.exhausted() && cursor.current_doc_id() == pivot_doc_id {
                        if let Some(entry) = cursor.current_entry() {
                            score += cursor.query_weight * entry.weight;
                        }
                    }
                }

                if score > threshold || heap.len() < top_k {
                    heap.push(SearchResult::new(PointId::new(pivot_doc_id), score));
                    if heap.len() > top_k {
                        heap.pop();
                    }
                    if heap.len() == top_k {
                        threshold = heap.peek().map_or(0.0, |r| r.score);
                    }
                }

                // Advance all cursors past this document
                for cursor in &mut cursors {
                    if !cursor.exhausted() && cursor.current_doc_id() == pivot_doc_id {
                        cursor.advance();
                    }
                }
            } else {
                // Advance cursors before pivot to pivot_doc_id
                for cursor in &mut cursors[..pivot_idx] {
                    if !cursor.exhausted() {
                        cursor.advance_to(pivot_doc_id);
                    }
                }
            }

            // Remove exhausted cursors
            cursors.retain(|c| !c.exhausted());
            if cursors.is_empty() {
                break;
            }
        }

        // Extract results from heap and sort by score descending
        let mut results: Vec<SearchResult> = heap.into_vec();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        Ok(results)
    }

    /// Search using MaxScore algorithm (optimized WAND variant).
    ///
    /// Similar to WAND but more aggressive at skipping low-scoring documents.
    pub fn search_maxscore(
        &self,
        collection: &str,
        vector_name: &str,
        query: &[(u32, f32)],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, VectorError> {
        // For now, delegate to WAND. MaxScore can be implemented later
        // as an optimization with block-max indices.
        self.search_wand(collection, vector_name, query, top_k)
    }
}

/// Cursor for WAND algorithm traversal.
struct WandCursor {
    posting_list: PostingList,
    position: usize,
    query_weight: f32,
    upper_bound: f32,
}

impl WandCursor {
    fn new(posting_list: PostingList, query_weight: f32, upper_bound: f32) -> Self {
        Self { posting_list, position: 0, query_weight, upper_bound }
    }

    fn exhausted(&self) -> bool {
        self.position >= self.posting_list.len()
    }

    fn current_doc_id(&self) -> u64 {
        if self.exhausted() {
            u64::MAX
        } else {
            self.posting_list.entries()[self.position].point_id.as_u64()
        }
    }

    fn current_entry(&self) -> Option<&PostingEntry> {
        if self.exhausted() {
            None
        } else {
            Some(&self.posting_list.entries()[self.position])
        }
    }

    fn advance(&mut self) {
        if !self.exhausted() {
            self.position += 1;
        }
    }

    fn advance_to(&mut self, doc_id: u64) {
        while !self.exhausted() && self.current_doc_id() < doc_id {
            self.position += 1;
        }
    }
}

/// Compute BM25 term score.
fn compute_bm25_term_score(
    query_weight: f32,
    doc_weight: f32,
    meta: Option<&InvertedIndexMeta>,
    _token_id: u32,
    df: usize,
    k1: f32,
    b: f32,
) -> f32 {
    let meta = match meta {
        Some(m) => m,
        None => return query_weight * doc_weight, // Fallback to dot product
    };

    if meta.doc_count == 0 {
        return 0.0;
    }

    // IDF component: log((N - df + 0.5) / (df + 0.5))
    let n = meta.doc_count as f32;
    let df = df as f32;
    let idf = ((n - df + 0.5) / (df + 0.5)).ln_1p();

    // TF component with length normalization
    // For sparse vectors, we use the weight as a proxy for term frequency
    let tf = doc_weight;
    let avg_dl = meta.avg_doc_length.max(1.0);
    // Assume document length is proportional to the weight
    let dl = doc_weight;

    let tf_component = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / avg_dl)));

    query_weight * idf * tf_component
}

/// Encode token IDs to bytes.
fn encode_token_ids(token_ids: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + token_ids.len() * 4);
    bytes.extend_from_slice(&(token_ids.len() as u32).to_le_bytes());
    for &token_id in token_ids {
        bytes.extend_from_slice(&token_id.to_le_bytes());
    }
    bytes
}

/// Decode token IDs from bytes.
fn decode_token_ids(bytes: &[u8]) -> Result<Vec<u32>, VectorError> {
    if bytes.len() < 4 {
        return Err(VectorError::Encoding("token ids too short".to_string()));
    }

    let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let expected_len = 4 + count * 4;

    if bytes.len() != expected_len {
        return Err(VectorError::Encoding(format!(
            "token ids length mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        )));
    }

    let mut token_ids = Vec::with_capacity(count);
    for i in 0..count {
        let offset = 4 + i * 4;
        let token_id = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        token_ids.push(token_id);
    }

    Ok(token_ids)
}

/// Calculate the next prefix for range scanning.
fn next_prefix(prefix: &[u8]) -> Vec<u8> {
    let mut result = prefix.to_vec();

    for byte in result.iter_mut().rev() {
        if *byte < 0xFF {
            *byte += 1;
            return result;
        }
    }

    result.push(0xFF);
    result
}

/// Delete all keys matching a prefix.
fn delete_by_prefix<T: Transaction>(
    tx: &mut T,
    table: &str,
    prefix: &[u8],
) -> Result<(), VectorError> {
    let prefix_end = next_prefix(prefix);

    let mut keys_to_delete = Vec::new();
    {
        let mut cursor =
            tx.range(table, Bound::Included(prefix), Bound::Excluded(prefix_end.as_slice()))?;

        while let Some((key, _)) = cursor.next()? {
            keys_to_delete.push(key);
        }
    }

    for key in keys_to_delete {
        tx.delete(table, &key)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifoldb_storage::backends::RedbEngine;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn create_test_index() -> InvertedIndex<RedbEngine> {
        let engine = RedbEngine::in_memory().unwrap();
        InvertedIndex::new(engine)
    }

    fn unique_name(prefix: &str) -> String {
        let count = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        format!("{}_{}", prefix, count)
    }

    #[test]
    fn posting_list_roundtrip() {
        let mut list = PostingList::new();
        list.add(PostingEntry::new(PointId::new(1), 0.5));
        list.add(PostingEntry::new(PointId::new(3), 0.3));
        list.add(PostingEntry::new(PointId::new(2), 0.8));

        let bytes = list.to_bytes();
        let restored = PostingList::from_bytes(&bytes).unwrap();

        assert_eq!(restored.len(), 3);
        assert!((restored.max_weight() - 0.8).abs() < 1e-6);

        // Should be sorted by point_id
        assert_eq!(restored.entries()[0].point_id, PointId::new(1));
        assert_eq!(restored.entries()[1].point_id, PointId::new(2));
        assert_eq!(restored.entries()[2].point_id, PointId::new(3));
    }

    #[test]
    fn posting_list_remove() {
        let mut list = PostingList::new();
        list.add(PostingEntry::new(PointId::new(1), 0.5));
        list.add(PostingEntry::new(PointId::new(2), 0.8));
        list.add(PostingEntry::new(PointId::new(3), 0.3));

        assert!(list.remove(PointId::new(2)));
        assert_eq!(list.len(), 2);
        assert!((list.max_weight() - 0.5).abs() < 1e-6);

        assert!(!list.remove(PointId::new(2))); // Already removed
    }

    #[test]
    fn insert_and_search() {
        let index = create_test_index();
        let collection = unique_name("collection");
        let vector = "keywords";

        // Insert some documents
        index.insert(&collection, vector, PointId::new(1), &[(100, 0.5), (200, 0.3)]).unwrap();
        index.insert(&collection, vector, PointId::new(2), &[(100, 0.8), (300, 0.2)]).unwrap();
        index.insert(&collection, vector, PointId::new(3), &[(200, 0.6), (300, 0.4)]).unwrap();

        // Search with DAAT
        let query = vec![(100, 1.0), (200, 0.5)];
        let results = index
            .search_daat(&collection, vector, &query, 10, ScoringFunction::DotProduct)
            .unwrap();

        assert!(!results.is_empty());
        // Point 1 has score: 0.5*1.0 + 0.3*0.5 = 0.65
        // Point 2 has score: 0.8*1.0 = 0.8
        // Point 3 has score: 0.6*0.5 = 0.3

        assert_eq!(results[0].point_id, PointId::new(2)); // Highest score
    }

    #[test]
    fn delete_document() {
        let index = create_test_index();
        let collection = unique_name("collection");
        let vector = "keywords";

        index.insert(&collection, vector, PointId::new(1), &[(100, 0.5)]).unwrap();
        index.insert(&collection, vector, PointId::new(2), &[(100, 0.8)]).unwrap();

        // Verify both exist
        let results = index
            .search_daat(&collection, vector, &[(100, 1.0)], 10, ScoringFunction::DotProduct)
            .unwrap();
        assert_eq!(results.len(), 2);

        // Delete one
        assert!(index.delete(&collection, vector, PointId::new(1)).unwrap());

        // Verify only one remains
        let results = index
            .search_daat(&collection, vector, &[(100, 1.0)], 10, ScoringFunction::DotProduct)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].point_id, PointId::new(2));
    }

    #[test]
    fn update_document() {
        let index = create_test_index();
        let collection = unique_name("collection");
        let vector = "keywords";

        index.insert(&collection, vector, PointId::new(1), &[(100, 0.5)]).unwrap();

        // Update with new vector
        index.update(&collection, vector, PointId::new(1), &[(200, 0.9)]).unwrap();

        // Old token should not match
        let results = index
            .search_daat(&collection, vector, &[(100, 1.0)], 10, ScoringFunction::DotProduct)
            .unwrap();
        assert!(results.is_empty());

        // New token should match
        let results = index
            .search_daat(&collection, vector, &[(200, 1.0)], 10, ScoringFunction::DotProduct)
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn wand_search() {
        let index = create_test_index();
        let collection = unique_name("collection");
        let vector = "keywords";

        // Insert many documents with 2 tokens to make WAND more effective
        for i in 0..100 {
            let weight = (i as f32 + 1.0) / 100.0;
            index
                .insert(&collection, vector, PointId::new(i), &[(100, weight), (200, weight * 0.5)])
                .unwrap();
        }

        // WAND search for top 5
        let results = index.search_wand(&collection, vector, &[(100, 1.0), (200, 0.5)], 5).unwrap();

        assert_eq!(results.len(), 5);
        // Results should be sorted by score descending
        for i in 0..4 {
            assert!(
                results[i].score >= results[i + 1].score,
                "Results should be sorted by score: {} >= {}",
                results[i].score,
                results[i + 1].score
            );
        }

        // Compare with DAAT to ensure correctness
        let daat_results = index
            .search_daat(
                &collection,
                vector,
                &[(100, 1.0), (200, 0.5)],
                5,
                ScoringFunction::DotProduct,
            )
            .unwrap();

        // Both should return the same result set (same point IDs in same order)
        assert_eq!(results.len(), daat_results.len());
        for (wand_r, daat_r) in results.iter().zip(daat_results.iter()) {
            assert_eq!(wand_r.point_id, daat_r.point_id);
            assert!((wand_r.score - daat_r.score).abs() < 1e-5);
        }
    }

    #[test]
    fn metadata_tracking() {
        let index = create_test_index();
        let collection = unique_name("collection");
        let vector = "keywords";

        index.insert(&collection, vector, PointId::new(1), &[(100, 0.5), (200, 0.3)]).unwrap();
        index.insert(&collection, vector, PointId::new(2), &[(100, 0.8)]).unwrap();

        let meta = index.get_meta(&collection, vector).unwrap();
        assert_eq!(meta.doc_count, 2);
        assert_eq!(meta.total_tokens, 3);
        assert!((meta.avg_doc_length - 1.5).abs() < 0.01);

        index.delete(&collection, vector, PointId::new(1)).unwrap();

        let meta = index.get_meta(&collection, vector).unwrap();
        assert_eq!(meta.doc_count, 1);
        assert_eq!(meta.total_tokens, 1);
    }

    #[test]
    fn bm25_scoring() {
        let index = create_test_index();
        let collection = unique_name("collection");
        let vector = "keywords";

        index.insert(&collection, vector, PointId::new(1), &[(100, 0.5)]).unwrap();
        index.insert(&collection, vector, PointId::new(2), &[(100, 0.8)]).unwrap();

        let results = index
            .search_daat(&collection, vector, &[(100, 1.0)], 10, ScoringFunction::bm25())
            .unwrap();

        assert_eq!(results.len(), 2);
        // BM25 should still rank doc 2 higher due to higher weight
        assert_eq!(results[0].point_id, PointId::new(2));
    }

    #[test]
    fn empty_query() {
        let index = create_test_index();
        let collection = unique_name("collection");
        let vector = "keywords";

        index.insert(&collection, vector, PointId::new(1), &[(100, 0.5)]).unwrap();

        let results =
            index.search_daat(&collection, vector, &[], 10, ScoringFunction::DotProduct).unwrap();
        assert!(results.is_empty());

        let results = index.search_wand(&collection, vector, &[], 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn no_matching_tokens() {
        let index = create_test_index();
        let collection = unique_name("collection");
        let vector = "keywords";

        index.insert(&collection, vector, PointId::new(1), &[(100, 0.5)]).unwrap();

        // Query for a token that doesn't exist
        let results = index
            .search_daat(&collection, vector, &[(999, 1.0)], 10, ScoringFunction::DotProduct)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn delete_vector_index() {
        let index = create_test_index();
        let collection = unique_name("collection");

        index.insert(&collection, "v1", PointId::new(1), &[(100, 0.5)]).unwrap();
        index.insert(&collection, "v2", PointId::new(1), &[(100, 0.8)]).unwrap();

        // Delete v1 index
        index.delete_vector(&collection, "v1").unwrap();

        // v1 should be gone
        assert!(index.get_meta(&collection, "v1").is_err());

        // v2 should still exist
        assert!(index.get_meta(&collection, "v2").is_ok());
    }
}

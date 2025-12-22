//! Product Quantization implementation.
//!
//! Product Quantization compresses vectors by splitting them into subspaces
//! and quantizing each subspace independently.

use crate::distance::DistanceMetric;
use crate::error::VectorError;

use super::config::PQConfig;
use super::training::{KMeans, KMeansConfig};

/// A compressed vector code from Product Quantization.
///
/// Each byte represents an index into the corresponding subspace codebook.
/// For 256 centroids (default), this is one byte per segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PQCode {
    /// The centroid indices, one per segment.
    codes: Vec<u8>,
    /// Number of bits per code (for serialization).
    bits_per_code: u8,
}

impl PQCode {
    /// Create a new PQ code from raw indices.
    ///
    /// # Arguments
    ///
    /// - `codes`: Centroid indices, one per segment
    /// - `bits_per_code`: Number of bits per index (typically 8 for 256 centroids)
    #[must_use]
    pub fn new(codes: Vec<u8>, bits_per_code: u8) -> Self {
        Self { codes, bits_per_code }
    }

    /// Get the number of segments.
    #[must_use]
    pub fn num_segments(&self) -> usize {
        self.codes.len()
    }

    /// Get the code (centroid index) for a segment.
    #[must_use]
    pub fn get(&self, segment: usize) -> Option<u8> {
        self.codes.get(segment).copied()
    }

    /// Get all codes as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.codes
    }

    /// Convert to bytes for storage.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        // For 8-bit codes, just return the codes directly
        // For other bit widths, we'd need packing
        if self.bits_per_code == 8 {
            self.codes.clone()
        } else {
            // Pack codes into bytes
            self.pack_codes()
        }
    }

    /// Create from bytes.
    ///
    /// # Arguments
    ///
    /// - `bytes`: Packed code bytes
    /// - `num_segments`: Number of segments (codes)
    /// - `bits_per_code`: Number of bits per code
    #[must_use]
    pub fn from_bytes(bytes: &[u8], num_segments: usize, bits_per_code: u8) -> Self {
        if bits_per_code == 8 {
            Self { codes: bytes[..num_segments].to_vec(), bits_per_code }
        } else {
            Self::unpack_codes(bytes, num_segments, bits_per_code)
        }
    }

    /// Pack codes into bytes (for non-8-bit codes).
    fn pack_codes(&self) -> Vec<u8> {
        let total_bits = self.codes.len() * self.bits_per_code as usize;
        let num_bytes = total_bits.div_ceil(8);
        let mut bytes = vec![0u8; num_bytes];

        let mut bit_pos = 0usize;
        for &code in &self.codes {
            let byte_idx = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            // Write lower bits to current byte
            bytes[byte_idx] |= code << bit_offset;

            // Handle codes that span byte boundaries
            if bit_offset + self.bits_per_code as usize > 8 && byte_idx + 1 < bytes.len() {
                bytes[byte_idx + 1] |= code >> (8 - bit_offset);
            }

            bit_pos += self.bits_per_code as usize;
        }

        bytes
    }

    /// Unpack codes from bytes.
    fn unpack_codes(bytes: &[u8], num_segments: usize, bits_per_code: u8) -> Self {
        let mask = (1u8 << bits_per_code) - 1;
        let mut codes = Vec::with_capacity(num_segments);

        let mut bit_pos = 0usize;
        for _ in 0..num_segments {
            let byte_idx = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            let code = if bit_offset + bits_per_code as usize <= 8 {
                (bytes[byte_idx] >> bit_offset) & mask
            } else {
                let low = bytes[byte_idx] >> bit_offset;
                let high = if byte_idx + 1 < bytes.len() {
                    bytes[byte_idx + 1] << (8 - bit_offset)
                } else {
                    0
                };
                (low | high) & mask
            };

            codes.push(code);
            bit_pos += bits_per_code as usize;
        }

        Self { codes, bits_per_code }
    }
}

/// Product Quantizer for vector compression.
///
/// The quantizer stores codebooks (centroids) for each subspace and provides
/// methods for encoding vectors and computing approximate distances.
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    /// Configuration.
    config: PQConfig,
    /// Codebooks: `codebooks[segment][centroid_idx]` = centroid vector.
    codebooks: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    /// Train a Product Quantizer on training data.
    ///
    /// # Arguments
    ///
    /// - `config`: PQ configuration
    /// - `training_data`: Training vectors (must all have dimension == config.dimension)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Configuration is invalid
    /// - Training data is empty
    /// - Training vectors have wrong dimension
    pub fn train(config: &PQConfig, training_data: &[&[f32]]) -> Result<Self, VectorError> {
        config.validate()?;

        if training_data.is_empty() {
            return Err(VectorError::Encoding("cannot train PQ on empty data".to_string()));
        }

        // Validate dimensions
        for (i, v) in training_data.iter().enumerate() {
            if v.len() != config.dimension {
                return Err(VectorError::DimensionMismatch {
                    expected: config.dimension,
                    actual: v.len(),
                });
            }
            if i > 1000 {
                break; // Only check first 1000 for performance
            }
        }

        let subspace_dim = config.subspace_dimension();
        let mut codebooks = Vec::with_capacity(config.num_segments);

        // Train a codebook for each segment
        for segment in 0..config.num_segments {
            let start = segment * subspace_dim;
            let end = start + subspace_dim;

            // Extract subvectors for this segment
            let subvectors: Vec<Vec<f32>> =
                training_data.iter().map(|v| v[start..end].to_vec()).collect();

            let subvector_refs: Vec<&[f32]> = subvectors.iter().map(|v| v.as_slice()).collect();

            // Train k-means on this segment
            let kmeans_config = KMeansConfig::new(config.num_centroids)
                .with_max_iterations(config.training_iterations)
                .with_seed(config.seed.map(|s| s + segment as u64).unwrap_or(segment as u64));

            let kmeans = KMeans::train(&subvector_refs, &kmeans_config, config.distance_metric)?;
            codebooks.push(kmeans.centroids);
        }

        Ok(Self { config: config.clone(), codebooks })
    }

    /// Create a Product Quantizer from pre-trained codebooks.
    ///
    /// # Arguments
    ///
    /// - `config`: PQ configuration
    /// - `codebooks`: Pre-trained codebooks, shape `[num_segments][num_centroids][subspace_dim]`
    ///
    /// # Errors
    ///
    /// Returns an error if codebooks don't match the configuration.
    pub fn from_codebooks(
        config: &PQConfig,
        codebooks: Vec<Vec<Vec<f32>>>,
    ) -> Result<Self, VectorError> {
        config.validate()?;

        if codebooks.len() != config.num_segments {
            return Err(VectorError::Encoding(format!(
                "expected {} codebooks, got {}",
                config.num_segments,
                codebooks.len()
            )));
        }

        let subspace_dim = config.subspace_dimension();
        for (i, codebook) in codebooks.iter().enumerate() {
            if codebook.len() != config.num_centroids {
                return Err(VectorError::Encoding(format!(
                    "codebook {} has {} centroids, expected {}",
                    i,
                    codebook.len(),
                    config.num_centroids
                )));
            }
            for centroid in codebook {
                if centroid.len() != subspace_dim {
                    return Err(VectorError::DimensionMismatch {
                        expected: subspace_dim,
                        actual: centroid.len(),
                    });
                }
            }
        }

        Ok(Self { config: config.clone(), codebooks })
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &PQConfig {
        &self.config
    }

    /// Get the codebooks.
    #[must_use]
    pub fn codebooks(&self) -> &[Vec<Vec<f32>>] {
        &self.codebooks
    }

    /// Encode a vector into a PQ code.
    ///
    /// # Arguments
    ///
    /// - `vector`: Input vector with dimension == config.dimension
    ///
    /// # Panics
    ///
    /// Panics if the vector has wrong dimension.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn encode(&self, vector: &[f32]) -> PQCode {
        debug_assert_eq!(vector.len(), self.config.dimension);

        let subspace_dim = self.config.subspace_dimension();
        let mut codes = Vec::with_capacity(self.config.num_segments);

        for (segment, codebook) in self.codebooks.iter().enumerate() {
            let start = segment * subspace_dim;
            let end = start + subspace_dim;
            let subvector = &vector[start..end];

            // Find nearest centroid
            let mut min_dist = f32::MAX;
            let mut min_idx = 0u8;

            for (idx, centroid) in codebook.iter().enumerate() {
                let dist = self.subspace_distance(subvector, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = idx as u8;
                }
            }

            codes.push(min_idx);
        }

        PQCode::new(codes, self.config.bits_per_code() as u8)
    }

    /// Decode a PQ code back to an approximate vector.
    ///
    /// The reconstructed vector is the concatenation of the centroids
    /// indicated by the code.
    #[must_use]
    pub fn decode(&self, code: &PQCode) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.config.dimension);

        for (segment, &idx) in code.as_slice().iter().enumerate() {
            let centroid = &self.codebooks[segment][idx as usize];
            vector.extend_from_slice(centroid);
        }

        vector
    }

    /// Compute a distance lookup table for asymmetric distance computation (ADC).
    ///
    /// The table contains precomputed distances from each subvector of the query
    /// to all centroids in the corresponding codebook.
    ///
    /// Shape: `table[segment][centroid_idx]` = distance
    ///
    /// # Arguments
    ///
    /// - `query`: Query vector with dimension == config.dimension
    #[must_use]
    pub fn compute_distance_table(&self, query: &[f32]) -> DistanceTable {
        debug_assert_eq!(query.len(), self.config.dimension);

        let subspace_dim = self.config.subspace_dimension();
        let mut table = Vec::with_capacity(self.config.num_segments);

        for (segment, codebook) in self.codebooks.iter().enumerate() {
            let start = segment * subspace_dim;
            let end = start + subspace_dim;
            let subvector = &query[start..end];

            let mut segment_distances = Vec::with_capacity(codebook.len());
            for centroid in codebook {
                let dist = self.subspace_distance(subvector, centroid);
                segment_distances.push(dist);
            }

            table.push(segment_distances);
        }

        DistanceTable { table, metric: self.config.distance_metric }
    }

    /// Compute asymmetric distance from a precomputed distance table to a PQ code.
    ///
    /// This is the primary method for fast approximate nearest neighbor search.
    /// The query vector is exact, while the database vector is compressed.
    ///
    /// # Arguments
    ///
    /// - `table`: Distance table from `compute_distance_table`
    /// - `code`: PQ code of a database vector
    #[must_use]
    #[inline]
    pub fn asymmetric_distance(&self, table: &DistanceTable, code: &PQCode) -> f32 {
        let mut total = 0.0f32;

        for (segment, &idx) in code.as_slice().iter().enumerate() {
            total += table.table[segment][idx as usize];
        }

        // For Euclidean distance, we sum squared distances and take sqrt at the end
        // For other metrics, we just sum
        match self.config.distance_metric {
            DistanceMetric::Euclidean => total.sqrt(),
            _ => total,
        }
    }

    /// Compute asymmetric squared distance (faster, no sqrt).
    ///
    /// For Euclidean distance, returns the squared distance.
    /// For other metrics, returns the same as `asymmetric_distance`.
    #[must_use]
    #[inline]
    pub fn asymmetric_distance_squared(&self, table: &DistanceTable, code: &PQCode) -> f32 {
        let mut total = 0.0f32;

        for (segment, &idx) in code.as_slice().iter().enumerate() {
            total += table.table[segment][idx as usize];
        }

        total
    }

    /// Compute symmetric distance between two PQ codes.
    ///
    /// This is faster but less accurate than asymmetric distance.
    /// Both vectors are compressed.
    #[must_use]
    pub fn symmetric_distance(&self, code_a: &PQCode, code_b: &PQCode) -> f32 {
        let mut total = 0.0f32;

        for segment in 0..self.config.num_segments {
            let idx_a = code_a.as_slice()[segment] as usize;
            let idx_b = code_b.as_slice()[segment] as usize;

            let centroid_a = &self.codebooks[segment][idx_a];
            let centroid_b = &self.codebooks[segment][idx_b];

            total += self.subspace_distance(centroid_a, centroid_b);
        }

        match self.config.distance_metric {
            DistanceMetric::Euclidean => total.sqrt(),
            _ => total,
        }
    }

    /// Compute distance between two subvectors.
    #[inline]
    fn subspace_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.distance_metric {
            DistanceMetric::Euclidean => {
                // Return squared distance for efficiency (sqrt at the end)
                a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
            }
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0
                } else {
                    1.0 - (dot / (norm_a * norm_b))
                }
            }
            DistanceMetric::DotProduct => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                -dot
            }
            DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
            DistanceMetric::Chebyshev => {
                a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
            }
        }
    }

    /// Serialize the quantizer to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Version byte
        bytes.push(1u8);

        // Config
        bytes.extend_from_slice(&(self.config.dimension as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.num_segments as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.num_centroids as u32).to_le_bytes());
        bytes.push(match self.config.distance_metric {
            DistanceMetric::Euclidean => 0,
            DistanceMetric::Cosine => 1,
            DistanceMetric::DotProduct => 2,
            DistanceMetric::Manhattan => 3,
            DistanceMetric::Chebyshev => 4,
        });

        // Codebooks
        for codebook in &self.codebooks {
            for centroid in codebook {
                for &val in centroid {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
            }
        }

        bytes
    }

    /// Deserialize a quantizer from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are malformed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        if bytes.len() < 14 {
            return Err(VectorError::Encoding("PQ bytes too short".to_string()));
        }

        let version = bytes[0];
        if version != 1 {
            return Err(VectorError::Encoding(format!("unsupported PQ version: {}", version)));
        }

        let dimension = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
        let num_segments = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
        let num_centroids =
            u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]) as usize;
        let distance_metric = match bytes[13] {
            0 => DistanceMetric::Euclidean,
            1 => DistanceMetric::Cosine,
            2 => DistanceMetric::DotProduct,
            3 => DistanceMetric::Manhattan,
            4 => DistanceMetric::Chebyshev,
            m => return Err(VectorError::Encoding(format!("unknown metric: {}", m))),
        };

        let config = PQConfig::new(dimension, num_segments)
            .with_num_centroids(num_centroids)
            .with_distance_metric(distance_metric);

        let subspace_dim = dimension / num_segments;
        let codebook_size = num_centroids * subspace_dim * 4; // 4 bytes per f32
        let expected_size = 14 + num_segments * codebook_size;

        if bytes.len() < expected_size {
            return Err(VectorError::Encoding(format!(
                "PQ bytes too short: expected {}, got {}",
                expected_size,
                bytes.len()
            )));
        }

        let mut offset = 14;
        let mut codebooks = Vec::with_capacity(num_segments);

        for _ in 0..num_segments {
            let mut codebook = Vec::with_capacity(num_centroids);
            for _ in 0..num_centroids {
                let mut centroid = Vec::with_capacity(subspace_dim);
                for _ in 0..subspace_dim {
                    let val = f32::from_le_bytes([
                        bytes[offset],
                        bytes[offset + 1],
                        bytes[offset + 2],
                        bytes[offset + 3],
                    ]);
                    centroid.push(val);
                    offset += 4;
                }
                codebook.push(centroid);
            }
            codebooks.push(codebook);
        }

        Self::from_codebooks(&config, codebooks)
    }
}

/// Precomputed distance table for asymmetric distance computation.
///
/// Contains distances from query subvectors to all centroids in each codebook.
#[derive(Debug, Clone)]
pub struct DistanceTable {
    /// Distance table: `table[segment][centroid_idx]` = distance.
    table: Vec<Vec<f32>>,
    /// Distance metric used.
    metric: DistanceMetric,
}

impl DistanceTable {
    /// Get the distance for a segment and centroid index.
    #[must_use]
    #[inline]
    pub fn get(&self, segment: usize, centroid_idx: usize) -> f32 {
        self.table[segment][centroid_idx]
    }

    /// Get the number of segments.
    #[must_use]
    pub fn num_segments(&self) -> usize {
        self.table.len()
    }

    /// Get the number of centroids per segment.
    #[must_use]
    pub fn num_centroids(&self) -> usize {
        self.table.first().map_or(0, Vec::len)
    }

    /// Get the distance metric used.
    #[must_use]
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng_state = seed;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        rng_state ^= rng_state << 13;
                        rng_state ^= rng_state >> 7;
                        rng_state ^= rng_state << 17;
                        (rng_state as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_pq_code_roundtrip() {
        let code = PQCode::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 8);
        let bytes = code.to_bytes();
        let restored = PQCode::from_bytes(&bytes, 8, 8);
        assert_eq!(code, restored);
    }

    #[test]
    fn test_pq_code_4bit_roundtrip() {
        let code = PQCode::new(vec![1, 15, 8, 3], 4);
        let bytes = code.to_bytes();
        let restored = PQCode::from_bytes(&bytes, 4, 4);
        assert_eq!(code, restored);
    }

    #[test]
    fn test_pq_train_and_encode() {
        // Generate random training data
        let training_data = generate_random_vectors(100, 32, 42);
        let training_refs: Vec<&[f32]> = training_data.iter().map(|v| v.as_slice()).collect();

        let config = PQConfig::new(32, 4).with_num_centroids(16).with_seed(42);

        let pq = ProductQuantizer::train(&config, &training_refs).unwrap();

        // Encode a vector
        let vector = generate_random_vectors(1, 32, 123)[0].clone();
        let code = pq.encode(&vector);

        assert_eq!(code.num_segments(), 4);
        for i in 0..4 {
            assert!(code.get(i).unwrap() < 16);
        }
    }

    #[test]
    fn test_pq_decode() {
        let training_data = generate_random_vectors(100, 32, 42);
        let training_refs: Vec<&[f32]> = training_data.iter().map(|v| v.as_slice()).collect();

        let config = PQConfig::new(32, 4).with_num_centroids(16).with_seed(42);
        let pq = ProductQuantizer::train(&config, &training_refs).unwrap();

        let vector = generate_random_vectors(1, 32, 123)[0].clone();
        let code = pq.encode(&vector);
        let decoded = pq.decode(&code);

        assert_eq!(decoded.len(), 32);
    }

    #[test]
    fn test_asymmetric_distance() {
        let training_data = generate_random_vectors(200, 64, 42);
        let training_refs: Vec<&[f32]> = training_data.iter().map(|v| v.as_slice()).collect();

        let config = PQConfig::new(64, 8).with_num_centroids(32).with_seed(42);
        let pq = ProductQuantizer::train(&config, &training_refs).unwrap();

        // Encode database vectors
        let db_vectors = generate_random_vectors(50, 64, 100);
        let codes: Vec<PQCode> = db_vectors.iter().map(|v| pq.encode(v)).collect();

        // Query vector
        let query = generate_random_vectors(1, 64, 200)[0].clone();
        let table = pq.compute_distance_table(&query);

        // Compute approximate distances
        let approx_dists: Vec<f32> =
            codes.iter().map(|c| pq.asymmetric_distance(&table, c)).collect();

        // All distances should be non-negative for Euclidean
        for d in &approx_dists {
            assert!(*d >= 0.0, "distance should be non-negative: {}", d);
        }
    }

    #[test]
    fn test_symmetric_distance() {
        let training_data = generate_random_vectors(100, 32, 42);
        let training_refs: Vec<&[f32]> = training_data.iter().map(|v| v.as_slice()).collect();

        let config = PQConfig::new(32, 4).with_num_centroids(16).with_seed(42);
        let pq = ProductQuantizer::train(&config, &training_refs).unwrap();

        let v1 = generate_random_vectors(1, 32, 100)[0].clone();
        let v2 = generate_random_vectors(1, 32, 200)[0].clone();

        let code1 = pq.encode(&v1);
        let code2 = pq.encode(&v2);

        let dist = pq.symmetric_distance(&code1, &code2);
        assert!(dist >= 0.0);

        // Distance to self should be 0
        let self_dist = pq.symmetric_distance(&code1, &code1);
        assert!(self_dist < 1e-6);
    }

    #[test]
    fn test_pq_serialization() {
        let training_data = generate_random_vectors(100, 32, 42);
        let training_refs: Vec<&[f32]> = training_data.iter().map(|v| v.as_slice()).collect();

        let config = PQConfig::new(32, 4).with_num_centroids(16).with_seed(42);
        let pq = ProductQuantizer::train(&config, &training_refs).unwrap();

        let bytes = pq.to_bytes();
        let restored = ProductQuantizer::from_bytes(&bytes).unwrap();

        // Verify config matches
        assert_eq!(pq.config().dimension, restored.config().dimension);
        assert_eq!(pq.config().num_segments, restored.config().num_segments);
        assert_eq!(pq.config().num_centroids, restored.config().num_centroids);

        // Verify codebooks match
        for (seg, (orig, rest)) in
            pq.codebooks().iter().zip(restored.codebooks().iter()).enumerate()
        {
            for (cent, (o, r)) in orig.iter().zip(rest.iter()).enumerate() {
                for (dim, (&ov, &rv)) in o.iter().zip(r.iter()).enumerate() {
                    assert!(
                        (ov - rv).abs() < 1e-6,
                        "mismatch at seg={}, cent={}, dim={}: {} vs {}",
                        seg,
                        cent,
                        dim,
                        ov,
                        rv
                    );
                }
            }
        }
    }

    #[test]
    fn test_distance_approximation_quality() {
        // Test that PQ distances approximate true distances reasonably well
        let training_data = generate_random_vectors(500, 64, 42);
        let training_refs: Vec<&[f32]> = training_data.iter().map(|v| v.as_slice()).collect();

        let config = PQConfig::new(64, 8).with_num_centroids(256).with_seed(42);
        let pq = ProductQuantizer::train(&config, &training_refs).unwrap();

        // Test vectors
        let query = generate_random_vectors(1, 64, 100)[0].clone();
        let database = generate_random_vectors(100, 64, 200);

        let table = pq.compute_distance_table(&query);

        // Compute true and approximate distances
        let mut correlations = Vec::new();
        for db_vec in &database {
            let true_dist: f32 =
                query.iter().zip(db_vec.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();

            let code = pq.encode(db_vec);
            let approx_dist = pq.asymmetric_distance(&table, &code);

            // Track relative error
            if true_dist > 0.1 {
                let rel_error = (approx_dist - true_dist).abs() / true_dist;
                correlations.push(rel_error);
            }
        }

        // Average relative error should be reasonable (< 50% for this test)
        let avg_error: f32 = correlations.iter().sum::<f32>() / correlations.len() as f32;
        assert!(avg_error < 0.5, "average relative error too high: {}", avg_error);
    }
}

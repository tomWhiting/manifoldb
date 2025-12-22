//! Integration tests for Product Quantization with HNSW index.

use manifoldb_core::EntityId;
use manifoldb_storage::backends::RedbEngine;
use manifoldb_vector::{
    distance::DistanceMetric,
    index::{HnswConfig, HnswIndex, VectorIndex},
    quantization::{PQConfig, ProductQuantizer},
    types::Embedding,
};

/// Generate random vectors with a simple xorshift PRNG.
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
fn test_pq_basic_encode_decode() {
    // Test basic PQ encoding and decoding
    let dim = 64;
    let num_segments = 8;
    let num_centroids = 32;

    // Generate training data
    let training_data = generate_random_vectors(200, dim, 42);
    let training_refs: Vec<&[f32]> = training_data.iter().map(|v| v.as_slice()).collect();

    // Train PQ
    let config = PQConfig::new(dim, num_segments).with_num_centroids(num_centroids).with_seed(42);

    let pq = ProductQuantizer::train(&config, &training_refs).unwrap();

    // Encode a test vector
    let test_vector = generate_random_vectors(1, dim, 100)[0].clone();
    let code = pq.encode(&test_vector);

    assert_eq!(code.num_segments(), num_segments);

    // Decode and verify dimension
    let decoded = pq.decode(&code);
    assert_eq!(decoded.len(), dim);

    // Verify reconstruction error is bounded
    let reconstruction_error: f32 =
        test_vector.iter().zip(decoded.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();

    // Reconstruction should be within reasonable bounds
    // (exact value depends on data distribution and k)
    assert!(reconstruction_error < test_vector.iter().map(|x| x * x).sum::<f32>().sqrt() * 2.0);
}

#[test]
fn test_pq_adc_distance_approximation() {
    // Test that ADC distances correlate with true distances
    let dim = 64;
    let num_segments = 8;
    let num_centroids = 64;

    // Generate training data
    let training_data = generate_random_vectors(500, dim, 42);
    let training_refs: Vec<&[f32]> = training_data.iter().map(|v| v.as_slice()).collect();

    // Train PQ
    let config = PQConfig::new(dim, num_segments).with_num_centroids(num_centroids).with_seed(42);

    let pq = ProductQuantizer::train(&config, &training_refs).unwrap();

    // Generate query and database vectors
    let query = generate_random_vectors(1, dim, 100)[0].clone();
    let database = generate_random_vectors(100, dim, 200);

    // Compute distance table for ADC
    let table = pq.compute_distance_table(&query);

    // Compare true and approximate distances
    let mut true_distances = Vec::new();
    let mut approx_distances = Vec::new();

    for db_vec in &database {
        // True Euclidean distance
        let true_dist: f32 =
            query.iter().zip(db_vec.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();

        // Approximate distance using ADC
        let code = pq.encode(db_vec);
        let approx_dist = pq.asymmetric_distance(&table, &code);

        true_distances.push(true_dist);
        approx_distances.push(approx_dist);
    }

    // Compute Spearman rank correlation
    // Sort indices by true distance and check that approximate distances are similar
    let mut indices: Vec<usize> = (0..database.len()).collect();
    indices.sort_by(|&a, &b| true_distances[a].partial_cmp(&true_distances[b]).unwrap());

    let mut approx_indices: Vec<usize> = (0..database.len()).collect();
    approx_indices.sort_by(|&a, &b| approx_distances[a].partial_cmp(&approx_distances[b]).unwrap());

    // Top-10 recall should be good
    let top_k = 10;
    let true_top_k: std::collections::HashSet<_> = indices.iter().take(top_k).copied().collect();
    let approx_top_k: std::collections::HashSet<_> =
        approx_indices.iter().take(top_k).copied().collect();

    let recall = true_top_k.intersection(&approx_top_k).count() as f32 / top_k as f32;
    assert!(recall >= 0.5, "Top-10 recall should be at least 50%: {}", recall);
}

#[test]
fn test_pq_serialization() {
    let dim = 32;
    let num_segments = 4;
    let num_centroids = 16;

    let training_data = generate_random_vectors(100, dim, 42);
    let training_refs: Vec<&[f32]> = training_data.iter().map(|v| v.as_slice()).collect();

    let config = PQConfig::new(dim, num_segments).with_num_centroids(num_centroids).with_seed(42);

    let pq = ProductQuantizer::train(&config, &training_refs).unwrap();

    // Serialize
    let bytes = pq.to_bytes();

    // Deserialize
    let pq2 = ProductQuantizer::from_bytes(&bytes).unwrap();

    // Verify they produce identical codes
    let test_vec = generate_random_vectors(1, dim, 100)[0].clone();
    let code1 = pq.encode(&test_vec);
    let code2 = pq2.encode(&test_vec);

    assert_eq!(code1.as_slice(), code2.as_slice());
}

#[test]
fn test_hnsw_with_pq_config() {
    // Test that HNSW index correctly stores and retrieves PQ configuration
    let engine = RedbEngine::in_memory().unwrap();
    let config = HnswConfig::new(16).with_pq(8).with_pq_centroids(256);

    let index =
        HnswIndex::new(engine, "test_pq", 64, DistanceMetric::Euclidean, config.clone()).unwrap();

    // Verify config is stored correctly
    assert!(index.config().pq_enabled());
    assert_eq!(index.config().pq_segments, 8);
    assert_eq!(index.config().pq_centroids, 256);
}

#[test]
fn test_hnsw_with_pq_persistence() {
    // Test that PQ config persists across index reload
    // Use a temp file for persistence testing
    let temp_dir = std::env::temp_dir().join(format!("pq_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();
    let db_path = temp_dir.join("test.db");

    let config = HnswConfig::new(16).with_pq(8).with_pq_centroids(512).with_ef_construction(100);

    // Create index with PQ enabled
    {
        let engine = RedbEngine::open(db_path.to_str().unwrap()).unwrap();
        let mut index =
            HnswIndex::new(engine, "test_pq_persist", 64, DistanceMetric::Cosine, config.clone())
                .unwrap();

        // Insert some vectors
        let vectors = generate_random_vectors(10, 64, 42);
        for (i, v) in vectors.iter().enumerate() {
            let embedding = Embedding::new(v.clone()).unwrap();
            index.insert(EntityId::new(i as u64 + 1), &embedding).unwrap();
        }
    }

    // Reopen and verify config
    {
        let engine = RedbEngine::open(db_path.to_str().unwrap()).unwrap();
        let index = HnswIndex::open(engine, "test_pq_persist").unwrap();

        assert!(index.config().pq_enabled());
        assert_eq!(index.config().pq_segments, 8);
        assert_eq!(index.config().pq_centroids, 512);
        assert_eq!(index.config().ef_construction, 100);
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_compression_ratio() {
    // Verify compression ratio calculations
    let config = PQConfig::new(128, 8).with_num_centroids(256);

    // Original: 128 * 4 = 512 bytes
    // Compressed: 8 bytes (8 segments * 1 byte each)
    let ratio = config.compression_ratio();
    assert!((ratio - 64.0).abs() < 0.01);

    // With 16-bit codes
    let config = PQConfig::new(128, 8).with_num_centroids(65536);
    let ratio = config.compression_ratio();
    assert!((ratio - 32.0).abs() < 0.01);
}

#[test]
fn test_pq_with_different_metrics() {
    let dim = 32;
    let num_segments = 4;

    let training_data = generate_random_vectors(100, dim, 42);
    let training_refs: Vec<&[f32]> = training_data.iter().map(|v| v.as_slice()).collect();

    // Test Euclidean
    let config = PQConfig::new(dim, num_segments)
        .with_distance_metric(DistanceMetric::Euclidean)
        .with_seed(42);
    let pq_euclidean = ProductQuantizer::train(&config, &training_refs).unwrap();

    // Test Cosine
    let config =
        PQConfig::new(dim, num_segments).with_distance_metric(DistanceMetric::Cosine).with_seed(42);
    let pq_cosine = ProductQuantizer::train(&config, &training_refs).unwrap();

    // Both should work
    let test_vec = generate_random_vectors(1, dim, 100)[0].clone();

    let code_euclidean = pq_euclidean.encode(&test_vec);
    let code_cosine = pq_cosine.encode(&test_vec);

    assert_eq!(code_euclidean.num_segments(), num_segments);
    assert_eq!(code_cosine.num_segments(), num_segments);
}

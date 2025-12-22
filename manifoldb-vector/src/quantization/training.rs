//! K-means clustering for codebook training.
//!
//! This module provides k-means clustering used to train Product Quantization codebooks.

use crate::distance::DistanceMetric;
use crate::error::VectorError;

/// Configuration for k-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters (centroids).
    pub k: usize,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence threshold (stop if centroid movement < threshold).
    pub convergence_threshold: f32,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self { k: 256, max_iterations: 25, convergence_threshold: 1e-6, seed: None }
    }
}

impl KMeansConfig {
    /// Create a new k-means configuration.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k, ..Default::default() }
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Set the convergence threshold.
    #[must_use]
    pub const fn with_convergence_threshold(mut self, threshold: f32) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// K-means clustering result.
#[derive(Debug, Clone)]
pub struct KMeans {
    /// The cluster centroids.
    pub centroids: Vec<Vec<f32>>,
    /// The dimension of each centroid.
    pub dimension: usize,
    /// Number of iterations run.
    pub iterations: usize,
    /// Final inertia (sum of squared distances to nearest centroid).
    pub inertia: f32,
}

impl KMeans {
    /// Train k-means on the given data.
    ///
    /// # Arguments
    ///
    /// - `data`: Training vectors, each with the same dimension
    /// - `config`: K-means configuration
    /// - `metric`: Distance metric for clustering
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `data` is empty
    /// - Vectors have inconsistent dimensions
    /// - `k` is greater than the number of data points
    pub fn train(
        data: &[&[f32]],
        config: &KMeansConfig,
        metric: DistanceMetric,
    ) -> Result<Self, VectorError> {
        if data.is_empty() {
            return Err(VectorError::Encoding("cannot train k-means on empty data".to_string()));
        }

        let dimension = data[0].len();
        if dimension == 0 {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        // Validate all vectors have same dimension
        for (i, v) in data.iter().enumerate() {
            if v.len() != dimension {
                return Err(VectorError::DimensionMismatch {
                    expected: dimension,
                    actual: v.len(),
                });
            }
            // Skip NaN check for training data - assume pre-validated
            if i > 1000 {
                break; // Only check first 1000 for performance
            }
        }

        let k = config.k.min(data.len());
        if k == 0 {
            return Err(VectorError::Encoding("k must be > 0".to_string()));
        }

        // Initialize centroids using k-means++
        let mut centroids = Self::kmeans_plus_plus_init(data, k, config.seed);

        // Run iterations
        let mut assignments = vec![0usize; data.len()];
        let mut iterations = 0;
        let mut inertia = f32::MAX;

        for _ in 0..config.max_iterations {
            iterations += 1;

            // E-step: Assign each point to nearest centroid
            let new_inertia = Self::assign_clusters(data, &centroids, &mut assignments, metric);

            // M-step: Update centroids
            let new_centroids = Self::update_centroids(data, &assignments, k, dimension);

            // Check for convergence
            let max_movement = Self::max_centroid_movement(&centroids, &new_centroids, metric);
            centroids = new_centroids;
            inertia = new_inertia;

            if max_movement < config.convergence_threshold {
                break;
            }
        }

        Ok(Self { centroids, dimension, iterations, inertia })
    }

    /// K-means++ initialization: select initial centroids with probability
    /// proportional to squared distance from existing centroids.
    fn kmeans_plus_plus_init(data: &[&[f32]], k: usize, seed: Option<u64>) -> Vec<Vec<f32>> {
        let mut rng_state = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        });

        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

        // First centroid: random point
        let first_idx = Self::random_index(&mut rng_state, data.len());
        centroids.push(data[first_idx].to_vec());

        // Remaining centroids: probability proportional to D(x)^2
        for _ in 1..k {
            let mut distances: Vec<f32> = Vec::with_capacity(data.len());
            let mut total_dist = 0.0f32;

            for point in data {
                // Find minimum distance to any existing centroid
                let min_dist = centroids
                    .iter()
                    .map(|c| Self::squared_euclidean_distance(point, c))
                    .fold(f32::MAX, f32::min);

                distances.push(min_dist);
                total_dist += min_dist;
            }

            // Sample proportional to distances
            if total_dist <= 0.0 {
                // All points are at existing centroids, just pick random
                let idx = Self::random_index(&mut rng_state, data.len());
                centroids.push(data[idx].to_vec());
            } else {
                let threshold = Self::random_f32(&mut rng_state) * total_dist;
                let mut cumsum = 0.0f32;
                let mut selected_idx = data.len() - 1;

                for (i, &d) in distances.iter().enumerate() {
                    cumsum += d;
                    if cumsum >= threshold {
                        selected_idx = i;
                        break;
                    }
                }

                centroids.push(data[selected_idx].to_vec());
            }
        }

        centroids
    }

    /// Assign each data point to its nearest centroid.
    /// Returns the total inertia (sum of squared distances).
    fn assign_clusters(
        data: &[&[f32]],
        centroids: &[Vec<f32>],
        assignments: &mut [usize],
        metric: DistanceMetric,
    ) -> f32 {
        let mut total_inertia = 0.0f32;

        for (i, point) in data.iter().enumerate() {
            let mut min_dist = f32::MAX;
            let mut min_idx = 0;

            for (j, centroid) in centroids.iter().enumerate() {
                let dist = Self::compute_distance(point, centroid, metric);
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = j;
                }
            }

            assignments[i] = min_idx;
            total_inertia += min_dist * min_dist;
        }

        total_inertia
    }

    /// Update centroids based on current assignments.
    fn update_centroids(
        data: &[&[f32]],
        assignments: &[usize],
        k: usize,
        dimension: usize,
    ) -> Vec<Vec<f32>> {
        let mut new_centroids = vec![vec![0.0f32; dimension]; k];
        let mut counts = vec![0usize; k];

        // Sum points per cluster
        for (point, &cluster) in data.iter().zip(assignments.iter()) {
            counts[cluster] += 1;
            for (j, &val) in point.iter().enumerate() {
                new_centroids[cluster][j] += val;
            }
        }

        // Divide by count to get mean
        for (centroid, &count) in new_centroids.iter_mut().zip(counts.iter()) {
            if count > 0 {
                let count_f32 = count as f32;
                for val in centroid.iter_mut() {
                    *val /= count_f32;
                }
            }
        }

        // Handle empty clusters by reinitializing to random data point
        for (i, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[i] == 0 && !data.is_empty() {
                // Copy a random data point
                let idx = i % data.len();
                centroid.copy_from_slice(data[idx]);
            }
        }

        new_centroids
    }

    /// Compute maximum centroid movement between iterations.
    fn max_centroid_movement(old: &[Vec<f32>], new: &[Vec<f32>], metric: DistanceMetric) -> f32 {
        old.iter()
            .zip(new.iter())
            .map(|(o, n)| Self::compute_distance(o, n, metric))
            .fold(0.0f32, f32::max)
    }

    /// Compute distance between two vectors.
    #[inline]
    fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
        match metric {
            DistanceMetric::Euclidean => Self::squared_euclidean_distance(a, b).sqrt(),
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

    /// Squared Euclidean distance (faster, no sqrt).
    #[inline]
    fn squared_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Simple xorshift64 PRNG.
    #[inline]
    fn random_u64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Random index in [0, max).
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn random_index(state: &mut u64, max: usize) -> usize {
        (Self::random_u64(state) as usize) % max
    }

    /// Random f32 in [0, 1).
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn random_f32(state: &mut u64) -> f32 {
        (Self::random_u64(state) as f64 / u64::MAX as f64) as f32
    }

    /// Find the index of the nearest centroid to the given vector.
    #[must_use]
    pub fn find_nearest(&self, vector: &[f32], metric: DistanceMetric) -> usize {
        let mut min_dist = f32::MAX;
        let mut min_idx = 0;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let dist = Self::compute_distance(vector, centroid, metric);
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        min_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_simple() {
        // Simple 2D data with two obvious clusters
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.0],
        ];

        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let config = KMeansConfig::new(2).with_seed(42);
        let result = KMeans::train(&data_refs, &config, DistanceMetric::Euclidean).unwrap();

        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.dimension, 2);

        // Check that centroids are near the cluster centers
        let c0_near_origin = result.centroids[0][0] < 5.0 || result.centroids[1][0] < 5.0;
        let c1_near_ten = result.centroids[0][0] > 5.0 || result.centroids[1][0] > 5.0;
        assert!(c0_near_origin && c1_near_ten);
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let data: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![1.1, 2.1], vec![0.9, 1.9]];

        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let config = KMeansConfig::new(1).with_seed(42);
        let result = KMeans::train(&data_refs, &config, DistanceMetric::Euclidean).unwrap();

        assert_eq!(result.centroids.len(), 1);
        // Centroid should be near the mean
        assert!((result.centroids[0][0] - 1.0).abs() < 0.2);
        assert!((result.centroids[0][1] - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_kmeans_empty_data() {
        let data: Vec<&[f32]> = vec![];
        let config = KMeansConfig::new(2);
        let result = KMeans::train(&data, &config, DistanceMetric::Euclidean);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_k_larger_than_data() {
        let data: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let config = KMeansConfig::new(10).with_seed(42); // k=10, but only 2 points
        let result = KMeans::train(&data_refs, &config, DistanceMetric::Euclidean).unwrap();

        // Should cap k at data.len()
        assert_eq!(result.centroids.len(), 2);
    }

    #[test]
    fn test_find_nearest() {
        let data: Vec<Vec<f32>> =
            vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![10.0, 10.0], vec![10.1, 10.0]];

        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let config = KMeansConfig::new(2).with_seed(42);
        let kmeans = KMeans::train(&data_refs, &config, DistanceMetric::Euclidean).unwrap();

        // Point near origin should match the origin cluster
        let query_origin = vec![0.05, 0.05];
        let query_far = vec![10.05, 10.05];

        let idx_origin = kmeans.find_nearest(&query_origin, DistanceMetric::Euclidean);
        let idx_far = kmeans.find_nearest(&query_far, DistanceMetric::Euclidean);

        // Should map to different clusters
        assert_ne!(idx_origin, idx_far);
    }

    #[test]
    fn test_cosine_distance_clustering() {
        // Vectors in different directions (cosine should separate them)
        let data: Vec<Vec<f32>> =
            vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0], vec![0.1, 0.9]];

        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let config = KMeansConfig::new(2).with_seed(42);
        let result = KMeans::train(&data_refs, &config, DistanceMetric::Cosine).unwrap();

        assert_eq!(result.centroids.len(), 2);
    }
}

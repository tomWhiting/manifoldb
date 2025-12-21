//! Test data generators for reproducible benchmarks and tests.
//!
//! All generators support seeded random generation for reproducibility.

use manifoldb::{Database, EntityId};

/// Simple pseudo-random number generator (Xorshift64)
/// for reproducible random numbers without external dependencies.
#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new RNG with the given seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x853c_49e6_748f_ea9b } else { seed } }
    }

    /// Generate next u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a random u64 in range [0, max).
    pub fn next_range(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        self.next_u64() % max
    }

    /// Generate a random f32 in range [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    /// Generate a random f32 in range [min, max).
    pub fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    /// Generate a random bool with given probability of true.
    pub fn next_bool(&mut self, probability: f32) -> bool {
        self.next_f32() < probability
    }
}

// ============================================================================
// Entity Generators
// ============================================================================

/// Generator for entities with various properties.
pub struct EntityGenerator {
    rng: Rng,
    labels: Vec<String>,
    string_pool: Vec<String>,
}

impl EntityGenerator {
    /// Create a new entity generator with the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            labels: vec![
                "Person".to_string(),
                "Document".to_string(),
                "Product".to_string(),
                "Event".to_string(),
                "Location".to_string(),
            ],
            string_pool: vec![
                "alpha".to_string(),
                "beta".to_string(),
                "gamma".to_string(),
                "delta".to_string(),
                "epsilon".to_string(),
                "zeta".to_string(),
                "eta".to_string(),
                "theta".to_string(),
            ],
        }
    }

    /// Generate a random label.
    pub fn random_label(&mut self) -> String {
        let idx = self.rng.next_range(self.labels.len() as u64) as usize;
        self.labels[idx].clone()
    }

    /// Generate a random string from pool.
    pub fn random_string(&mut self) -> String {
        let idx = self.rng.next_range(self.string_pool.len() as u64) as usize;
        self.string_pool[idx].clone()
    }

    /// Generate a random i64.
    pub fn random_int(&mut self) -> i64 {
        self.rng.next_u64() as i64
    }

    /// Generate a random i64 in range.
    pub fn random_int_range(&mut self, min: i64, max: i64) -> i64 {
        min + (self.rng.next_range((max - min) as u64) as i64)
    }

    /// Generate a random f64.
    pub fn random_float(&mut self) -> f64 {
        self.rng.next_f32() as f64
    }

    /// Generate a random embedding vector.
    pub fn random_embedding(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.rng.next_f32_range(-1.0, 1.0)).collect()
    }

    /// Generate entities and insert them into the database.
    ///
    /// Returns the generated entity IDs.
    pub fn generate_entities(&mut self, db: &Database, count: usize) -> Vec<EntityId> {
        let mut ids = Vec::with_capacity(count);

        let mut tx = db.begin().expect("failed to begin");
        for i in 0..count {
            let label = self.random_label();
            let name = format!("{}_{}", self.random_string(), i);
            let value = self.random_int_range(0, 1000);

            let entity = tx
                .create_entity()
                .expect("failed")
                .with_label(label)
                .with_property("name", name)
                .with_property("value", value);

            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }
        tx.commit().expect("failed");

        ids
    }

    /// Generate entities in batches for better performance at scale.
    pub fn generate_entities_batched(
        &mut self,
        db: &Database,
        count: usize,
        batch_size: usize,
    ) -> Vec<EntityId> {
        let mut all_ids = Vec::with_capacity(count);

        for batch_start in (0..count).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(count);
            let batch_count = batch_end - batch_start;

            let mut tx = db.begin().expect("failed to begin");
            for i in 0..batch_count {
                let label = self.random_label();
                let idx = batch_start + i;
                let entity = tx
                    .create_entity()
                    .expect("failed")
                    .with_label(label)
                    .with_property("index", idx as i64);

                all_ids.push(entity.id);
                tx.put_entity(&entity).expect("failed");
            }
            tx.commit().expect("failed");
        }

        all_ids
    }
}

// ============================================================================
// Graph Generators
// ============================================================================

/// Generator for various graph topologies.
pub struct GraphGenerator {
    rng: Rng,
}

impl GraphGenerator {
    /// Create a new graph generator with the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { rng: Rng::new(seed) }
    }

    /// Create a linear chain: 0 -> 1 -> 2 -> ... -> n-1
    pub fn create_linear_chain(&self, db: &Database, n: usize) -> Vec<EntityId> {
        let mut ids = Vec::with_capacity(n);

        let mut tx = db.begin().expect("failed");
        for i in 0..n {
            let entity = tx.create_entity().expect("failed").with_property("position", i as i64);
            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }

        for i in 0..(n - 1) {
            let edge = tx.create_edge(ids[i], ids[i + 1], "NEXT").expect("failed");
            tx.put_edge(&edge).expect("failed");
        }

        tx.commit().expect("failed");
        ids
    }

    /// Create a binary tree with given depth.
    pub fn create_binary_tree(&self, db: &Database, depth: usize) -> Vec<EntityId> {
        let n = (1 << (depth + 1)) - 1;
        let mut ids = Vec::with_capacity(n);

        let mut tx = db.begin().expect("failed");

        for i in 0..n {
            let entity = tx.create_entity().expect("failed").with_property("index", i as i64);
            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }

        for i in 0..n {
            let left = 2 * i + 1;
            let right = 2 * i + 2;

            if left < n {
                let edge = tx.create_edge(ids[i], ids[left], "LEFT").expect("failed");
                tx.put_edge(&edge).expect("failed");
            }
            if right < n {
                let edge = tx.create_edge(ids[i], ids[right], "RIGHT").expect("failed");
                tx.put_edge(&edge).expect("failed");
            }
        }

        tx.commit().expect("failed");
        ids
    }

    /// Create a random graph with n nodes and m edges.
    pub fn create_random_graph(&mut self, db: &Database, n: usize, m: usize) -> Vec<EntityId> {
        let mut ids = Vec::with_capacity(n);

        let mut tx = db.begin().expect("failed");

        for i in 0..n {
            let entity = tx.create_entity().expect("failed").with_property("id", i as i64);
            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }

        for _ in 0..m {
            let src = self.rng.next_range(n as u64) as usize;
            let mut dst = self.rng.next_range(n as u64) as usize;
            // Avoid self-loops
            while dst == src {
                dst = self.rng.next_range(n as u64) as usize;
            }

            let edge = tx.create_edge(ids[src], ids[dst], "CONNECTS").expect("failed");
            tx.put_edge(&edge).expect("failed");
        }

        tx.commit().expect("failed");
        ids
    }

    /// Create a star graph: center connected to n-1 leaves.
    pub fn create_star_graph(&self, db: &Database, n: usize) -> Vec<EntityId> {
        let mut ids = Vec::with_capacity(n);

        let mut tx = db.begin().expect("failed");

        // Center node
        let center = tx.create_entity().expect("failed").with_label("Center");
        ids.push(center.id);
        tx.put_entity(&center).expect("failed");

        // Leaf nodes
        for i in 1..n {
            let leaf = tx
                .create_entity()
                .expect("failed")
                .with_label("Leaf")
                .with_property("index", i as i64);
            ids.push(leaf.id);
            tx.put_entity(&leaf).expect("failed");

            let edge = tx.create_edge(center.id, leaf.id, "CONNECTS").expect("failed");
            tx.put_edge(&edge).expect("failed");
        }

        tx.commit().expect("failed");
        ids
    }

    /// Create a complete graph (all nodes connected to all other nodes).
    pub fn create_complete_graph(&self, db: &Database, n: usize) -> Vec<EntityId> {
        let mut ids = Vec::with_capacity(n);

        let mut tx = db.begin().expect("failed");

        for i in 0..n {
            let entity = tx.create_entity().expect("failed").with_property("id", i as i64);
            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let edge = tx.create_edge(ids[i], ids[j], "CONNECTED").expect("failed");
                tx.put_edge(&edge).expect("failed");
            }
        }

        tx.commit().expect("failed");
        ids
    }
}

// ============================================================================
// Social Network Generator
// ============================================================================

/// Generator for realistic social network graphs.
pub struct SocialNetworkGenerator {
    rng: Rng,
    names: Vec<String>,
}

impl SocialNetworkGenerator {
    /// Create a new social network generator.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            names: vec![
                "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy",
                "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Rose", "Sam",
                "Tina",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        }
    }

    /// Generate a social network with followers following a power-law distribution.
    ///
    /// Some users will have many followers, most will have few.
    pub fn generate_power_law_network(
        &mut self,
        db: &Database,
        user_count: usize,
        avg_follows: usize,
    ) -> Vec<EntityId> {
        let mut ids = Vec::with_capacity(user_count);

        let mut tx = db.begin().expect("failed");

        // Create users
        for i in 0..user_count {
            let name_idx = i % self.names.len();
            let name = format!("{}_{}", self.names[name_idx], i);

            let user = tx
                .create_entity()
                .expect("failed")
                .with_label("User")
                .with_property("name", name)
                .with_property("followers", 0i64)
                .with_property("following", 0i64);

            ids.push(user.id);
            tx.put_entity(&user).expect("failed");
        }

        // Create follows edges with power-law distribution
        // Higher indexed users are more likely to be followed
        let total_edges = user_count * avg_follows;
        for _ in 0..total_edges {
            let follower = self.rng.next_range(user_count as u64) as usize;

            // Power-law: prefer following users with lower index (more "popular")
            let popularity_factor = self.rng.next_f32();
            let target = (popularity_factor * popularity_factor * (user_count as f32)) as usize;
            let target = target.min(user_count - 1);

            if follower != target {
                let edge = tx.create_edge(ids[follower], ids[target], "FOLLOWS").expect("failed");
                tx.put_edge(&edge).expect("failed");
            }
        }

        tx.commit().expect("failed");
        ids
    }

    /// Generate friendship clusters (groups of friends that are densely connected).
    pub fn generate_clustered_network(
        &mut self,
        db: &Database,
        cluster_count: usize,
        users_per_cluster: usize,
        inter_cluster_edges: usize,
    ) -> Vec<EntityId> {
        let total_users = cluster_count * users_per_cluster;
        let mut ids = Vec::with_capacity(total_users);

        let mut tx = db.begin().expect("failed");

        // Create users in clusters
        for cluster in 0..cluster_count {
            for i in 0..users_per_cluster {
                let _user_id = cluster * users_per_cluster + i;
                let user = tx
                    .create_entity()
                    .expect("failed")
                    .with_label("User")
                    .with_property("cluster", cluster as i64)
                    .with_property("local_id", i as i64);

                ids.push(user.id);
                tx.put_entity(&user).expect("failed");
            }
        }

        // Dense intra-cluster connections
        for cluster in 0..cluster_count {
            let start = cluster * users_per_cluster;
            for i in 0..users_per_cluster {
                for j in (i + 1)..users_per_cluster {
                    if self.rng.next_bool(0.6) {
                        // 60% chance of connection within cluster
                        let edge = tx
                            .create_edge(ids[start + i], ids[start + j], "FRIENDS")
                            .expect("failed");
                        tx.put_edge(&edge).expect("failed");
                    }
                }
            }
        }

        // Sparse inter-cluster connections
        for _ in 0..inter_cluster_edges {
            let cluster_a = self.rng.next_range(cluster_count as u64) as usize;
            let mut cluster_b = self.rng.next_range(cluster_count as u64) as usize;
            while cluster_b == cluster_a {
                cluster_b = self.rng.next_range(cluster_count as u64) as usize;
            }

            let user_a = cluster_a * users_per_cluster
                + self.rng.next_range(users_per_cluster as u64) as usize;
            let user_b = cluster_b * users_per_cluster
                + self.rng.next_range(users_per_cluster as u64) as usize;

            let edge = tx.create_edge(ids[user_a], ids[user_b], "KNOWS").expect("failed");
            tx.put_edge(&edge).expect("failed");
        }

        tx.commit().expect("failed");
        ids
    }
}

// ============================================================================
// Vector Generators
// ============================================================================

/// Batch of vectors for testing.
pub struct VectorBatch {
    /// Entity IDs associated with vectors.
    pub entity_ids: Vec<u64>,
    /// The vectors themselves.
    pub vectors: Vec<Vec<f32>>,
    /// Dimension of vectors.
    pub dimension: usize,
}

impl VectorBatch {
    /// Create a new empty batch.
    #[must_use]
    pub const fn new(dimension: usize) -> Self {
        Self { entity_ids: Vec::new(), vectors: Vec::new(), dimension }
    }

    /// Get number of vectors in batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// Generator for random vectors.
pub struct RandomVectors {
    rng: Rng,
}

impl RandomVectors {
    /// Create a new vector generator with given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { rng: Rng::new(seed) }
    }

    /// Generate a random unit vector (normalized).
    pub fn unit_vector(&mut self, dim: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..dim).map(|_| self.rng.next_f32_range(-1.0, 1.0)).collect();

        // Normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.into_iter().map(|x| x / norm).collect()
        } else {
            // Edge case: all zeros, return unit vector in first dimension
            let mut result = vec![0.0; dim];
            result[0] = 1.0;
            result
        }
    }

    /// Generate a random vector in range [-1, 1].
    pub fn random_vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.rng.next_f32_range(-1.0, 1.0)).collect()
    }

    /// Generate a batch of random vectors.
    pub fn generate_batch(&mut self, count: usize, dim: usize) -> VectorBatch {
        let mut batch = VectorBatch::new(dim);

        for i in 0..count {
            batch.entity_ids.push((i + 1) as u64);
            batch.vectors.push(self.random_vector(dim));
        }

        batch
    }

    /// Generate a batch of unit vectors.
    pub fn generate_unit_batch(&mut self, count: usize, dim: usize) -> VectorBatch {
        let mut batch = VectorBatch::new(dim);

        for i in 0..count {
            batch.entity_ids.push((i + 1) as u64);
            batch.vectors.push(self.unit_vector(dim));
        }

        batch
    }

    /// Generate clustered vectors (vectors near cluster centers).
    pub fn generate_clustered(
        &mut self,
        count: usize,
        dim: usize,
        num_clusters: usize,
        cluster_radius: f32,
    ) -> VectorBatch {
        let mut batch = VectorBatch::new(dim);

        // Generate cluster centers
        let centers: Vec<Vec<f32>> = (0..num_clusters).map(|_| self.random_vector(dim)).collect();

        // Generate points around centers
        for i in 0..count {
            let center_idx = i % num_clusters;
            let center = &centers[center_idx];

            // Add noise to center
            let point: Vec<f32> = center
                .iter()
                .map(|&c| c + self.rng.next_f32_range(-cluster_radius, cluster_radius))
                .collect();

            batch.entity_ids.push((i + 1) as u64);
            batch.vectors.push(point);
        }

        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_reproducibility() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_entity_generator() {
        let db = Database::in_memory().expect("failed");
        let mut gen = EntityGenerator::new(42);

        let ids = gen.generate_entities(&db, 100);
        assert_eq!(ids.len(), 100);

        // Verify entities exist
        let tx = db.begin_read().expect("failed");
        for &id in &ids {
            assert!(tx.get_entity(id).expect("failed").is_some());
        }
    }

    #[test]
    fn test_graph_generator_chain() {
        let db = Database::in_memory().expect("failed");
        let gen = GraphGenerator::new(42);

        let ids = gen.create_linear_chain(&db, 10);
        assert_eq!(ids.len(), 10);

        let tx = db.begin_read().expect("failed");
        for i in 0..9 {
            let edges = tx.get_outgoing_edges(ids[i]).expect("failed");
            assert_eq!(edges.len(), 1);
            assert_eq!(edges[0].target, ids[i + 1]);
        }
    }

    #[test]
    fn test_random_vectors() {
        let mut gen = RandomVectors::new(42);

        let batch = gen.generate_batch(100, 128);
        assert_eq!(batch.len(), 100);
        assert_eq!(batch.dimension, 128);

        for v in &batch.vectors {
            assert_eq!(v.len(), 128);
        }
    }

    #[test]
    fn test_unit_vectors() {
        let mut gen = RandomVectors::new(42);

        for _ in 0..10 {
            let v = gen.unit_vector(100);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.001, "unit vector should have norm 1");
        }
    }
}

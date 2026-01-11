/**
 * Vector projection utilities for dimensionality reduction
 *
 * Provides UMAP, t-SNE, and PCA projections for visualizing high-dimensional
 * vector embeddings in 2D space.
 */
import * as druid from '@saehrimnir/druidjs'

export type ProjectionMethod = 'umap' | 'tsne' | 'pca'

export interface ProjectionConfig {
  method: ProjectionMethod
  // UMAP-specific parameters
  nNeighbors?: number
  minDist?: number
  // t-SNE specific parameters
  perplexity?: number
  // General parameters
  nComponents?: number
  nEpochs?: number
}

export interface ProjectedPoint {
  id: string
  x: number
  y: number
  originalVector?: number[]
}

export interface ProjectionResult {
  points: ProjectedPoint[]
  bounds: {
    minX: number
    maxX: number
    minY: number
    maxY: number
  }
}

export interface ClusterResult {
  clusters: Map<number, string[]>  // cluster ID -> point IDs
  labels: Map<string, number>      // point ID -> cluster ID
  centroids: Map<number, { x: number; y: number }>
}

const DEFAULT_CONFIG: ProjectionConfig = {
  method: 'pca',
  nNeighbors: 15,
  minDist: 0.1,
  perplexity: 30,
  nComponents: 2,
  nEpochs: 200,
}

/**
 * Project high-dimensional vectors to 2D using the specified method
 */
export function projectVectors(
  vectors: { id: string; vector: number[] }[],
  config: Partial<ProjectionConfig> = {}
): ProjectionResult {
  const cfg = { ...DEFAULT_CONFIG, ...config }

  if (vectors.length === 0) {
    return {
      points: [],
      bounds: { minX: 0, maxX: 0, minY: 0, maxY: 0 }
    }
  }

  if (vectors.length === 1) {
    return {
      points: [{ id: vectors[0].id, x: 0, y: 0, originalVector: vectors[0].vector }],
      bounds: { minX: 0, maxX: 0, minY: 0, maxY: 0 }
    }
  }

  // Create matrix from vectors
  const data = vectors.map(v => v.vector)
  const matrix = druid.Matrix.from(data)

  let projection: druid.Matrix

  switch (cfg.method) {
    case 'umap':
      projection = projectUMAP(matrix, cfg)
      break
    case 'tsne':
      projection = projectTSNE(matrix, cfg)
      break
    case 'pca':
    default:
      projection = projectPCA(matrix, cfg)
      break
  }

  // Convert projection to points
  const projectionData = projection.to2dArray as number[][]
  const points: ProjectedPoint[] = vectors.map((v, i) => ({
    id: v.id,
    x: projectionData[i]?.[0] ?? 0,
    y: projectionData[i]?.[1] ?? 0,
    originalVector: v.vector,
  }))

  // Calculate bounds
  let minX = Infinity, maxX = -Infinity
  let minY = Infinity, maxY = -Infinity

  for (const point of points) {
    minX = Math.min(minX, point.x)
    maxX = Math.max(maxX, point.x)
    minY = Math.min(minY, point.y)
    maxY = Math.max(maxY, point.y)
  }

  return {
    points,
    bounds: { minX, maxX, minY, maxY }
  }
}

function projectPCA(matrix: druid.Matrix, config: ProjectionConfig): druid.Matrix {
  const pca = new druid.PCA(matrix, config.nComponents ?? 2)
  return pca.transform()
}

function projectUMAP(matrix: druid.Matrix, config: ProjectionConfig): druid.Matrix {
  const umap = new druid.UMAP(matrix, {
    n_neighbors: config.nNeighbors ?? 15,
    min_dist: config.minDist ?? 0.1,
    d: config.nComponents ?? 2,
  })
  return umap.transform()
}

function projectTSNE(matrix: druid.Matrix, config: ProjectionConfig): druid.Matrix {
  const tsne = new druid.TSNE(matrix, {
    perplexity: config.perplexity ?? 30,
    d: config.nComponents ?? 2,
  })
  return tsne.transform()
}

/**
 * Calculate cosine similarity between two vectors
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same dimension')
  }

  let dotProduct = 0
  let normA = 0
  let normB = 0

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }

  normA = Math.sqrt(normA)
  normB = Math.sqrt(normB)

  if (normA === 0 || normB === 0) {
    return 0
  }

  return dotProduct / (normA * normB)
}

/**
 * Calculate euclidean distance between two vectors
 */
export function euclideanDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same dimension')
  }

  let sum = 0
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i]
    sum += diff * diff
  }

  return Math.sqrt(sum)
}

/**
 * Calculate similarity scores for all points relative to a reference point
 * Returns a map from point ID to similarity score (0-1)
 */
export function calculateSimilarities(
  referenceVector: number[],
  points: ProjectedPoint[],
  metric: 'cosine' | 'euclidean' = 'cosine'
): Map<string, number> {
  const similarities = new Map<string, number>()

  if (metric === 'cosine') {
    for (const point of points) {
      if (point.originalVector) {
        const similarity = cosineSimilarity(referenceVector, point.originalVector)
        // Normalize from [-1, 1] to [0, 1]
        similarities.set(point.id, (similarity + 1) / 2)
      }
    }
  } else {
    // For euclidean, we need to find the max distance to normalize
    let maxDistance = 0
    const distances = new Map<string, number>()

    for (const point of points) {
      if (point.originalVector) {
        const distance = euclideanDistance(referenceVector, point.originalVector)
        distances.set(point.id, distance)
        maxDistance = Math.max(maxDistance, distance)
      }
    }

    // Convert distances to similarities (1 = closest, 0 = farthest)
    for (const [id, distance] of distances) {
      similarities.set(id, maxDistance > 0 ? 1 - (distance / maxDistance) : 1)
    }
  }

  return similarities
}

/**
 * Cluster projected points using k-means
 */
export function clusterPoints(
  points: ProjectedPoint[],
  k: number = 5
): ClusterResult {
  if (points.length === 0 || k <= 0) {
    return {
      clusters: new Map(),
      labels: new Map(),
      centroids: new Map(),
    }
  }

  // Limit k to the number of points
  k = Math.min(k, points.length)

  // Create matrix from 2D projected coordinates
  const data = points.map(p => [p.x, p.y])
  const matrix = druid.Matrix.from(data)

  // Run k-means clustering
  const kmeans = new druid.KMeans(matrix, k)
  const clusterLabels = kmeans.get_clusters()

  // Build result structures
  const clusters = new Map<number, string[]>()
  const labels = new Map<string, number>()
  const centroids = new Map<number, { x: number; y: number }>()

  // Build point lookup map for O(1) access during centroid calculation
  const pointById = new Map<string, ProjectedPoint>()
  for (const point of points) {
    pointById.set(point.id, point)
  }

  // Assign points to clusters
  for (let i = 0; i < points.length; i++) {
    const clusterId = clusterLabels[i]
    const pointId = points[i].id

    labels.set(pointId, clusterId)

    if (!clusters.has(clusterId)) {
      clusters.set(clusterId, [])
    }
    clusters.get(clusterId)!.push(pointId)
  }

  // Calculate cluster centroids using O(1) lookups
  for (const [clusterId, memberIds] of clusters) {
    let sumX = 0, sumY = 0
    for (const memberId of memberIds) {
      const point = pointById.get(memberId)
      if (point) {
        sumX += point.x
        sumY += point.y
      }
    }
    centroids.set(clusterId, {
      x: sumX / memberIds.length,
      y: sumY / memberIds.length,
    })
  }

  return { clusters, labels, centroids }
}

/**
 * Generate a color based on similarity score
 * Similar (high score) -> warm colors (red/orange)
 * Dissimilar (low score) -> cool colors (blue)
 */
export function similarityToColor(similarity: number): string {
  // Clamp similarity to [0, 1]
  const s = Math.max(0, Math.min(1, similarity))

  // Interpolate hue from blue (240) to red (0)
  const hue = (1 - s) * 240

  return `hsl(${hue}, 70%, 55%)`
}

/**
 * Generate distinct colors for cluster IDs
 */
export function clusterToColor(clusterId: number, totalClusters: number): string {
  const hue = (clusterId * 360) / totalClusters
  return `hsl(${hue}, 65%, 55%)`
}

/**
 * Calculate convex hull for a set of points (for cluster boundaries)
 * Returns points in order for drawing
 */
export function convexHull(points: { x: number; y: number }[]): { x: number; y: number }[] {
  if (points.length < 3) {
    return points
  }

  // Find the point with lowest y (and leftmost if tie)
  let start = 0
  for (let i = 1; i < points.length; i++) {
    if (points[i].y < points[start].y ||
        (points[i].y === points[start].y && points[i].x < points[start].x)) {
      start = i
    }
  }

  // Sort points by polar angle with respect to start point
  const startPoint = points[start]
  const sorted = points
    .filter((_, i) => i !== start)
    .sort((a, b) => {
      const angleA = Math.atan2(a.y - startPoint.y, a.x - startPoint.x)
      const angleB = Math.atan2(b.y - startPoint.y, b.x - startPoint.x)
      if (angleA !== angleB) {
        return angleA - angleB
      }
      // If same angle, sort by distance
      const distA = (a.x - startPoint.x) ** 2 + (a.y - startPoint.y) ** 2
      const distB = (b.x - startPoint.x) ** 2 + (b.y - startPoint.y) ** 2
      return distA - distB
    })

  // Graham scan
  const hull: { x: number; y: number }[] = [startPoint]

  for (const point of sorted) {
    while (hull.length > 1) {
      const p1 = hull[hull.length - 2]
      const p2 = hull[hull.length - 1]
      const cross = (p2.x - p1.x) * (point.y - p1.y) - (p2.y - p1.y) * (point.x - p1.x)

      if (cross > 0) {
        break
      }
      hull.pop()
    }
    hull.push(point)
  }

  return hull
}

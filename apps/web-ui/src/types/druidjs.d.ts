/**
 * Type declarations for @saehrimnir/druidjs
 * DruidJS is a JavaScript library for dimensionality reduction
 */
declare module '@saehrimnir/druidjs' {
  export class Matrix {
    /**
     * Create a matrix from a 2D array
     */
    static from(data: number[][]): Matrix

    /**
     * Get the number of rows
     */
    get rows(): number

    /**
     * Get the number of columns
     */
    get cols(): number

    /**
     * Convert matrix to 2D array
     */
    get to2dArray(): number[][]

    /**
     * Get a specific entry
     */
    entry(i: number, j: number): number
  }

  export interface PCAOptions {
    seed?: number
  }

  export class PCA {
    constructor(matrix: Matrix, d?: number, options?: PCAOptions)
    transform(): Matrix
  }

  export interface UMAPOptions {
    n_neighbors?: number
    min_dist?: number
    d?: number
    local_connectivity?: number
    seed?: number
  }

  export class UMAP {
    constructor(matrix: Matrix, options?: UMAPOptions)
    transform(): Matrix
  }

  export interface TSNEOptions {
    perplexity?: number
    epsilon?: number
    d?: number
    seed?: number
  }

  export class TSNE {
    constructor(matrix: Matrix, options?: TSNEOptions)
    transform(): Matrix
  }

  export interface KMeansOptions {
    seed?: number
    max_iterations?: number
  }

  export class KMeans {
    constructor(matrix: Matrix, k: number, options?: KMeansOptions)
    /**
     * Get cluster labels for each point
     */
    get_clusters(): number[]
  }

  export class KMedoids {
    constructor(matrix: Matrix, k: number, options?: KMeansOptions)
    get_clusters(): number[]
  }

  export interface HierarchicalClusteringOptions {
    linkage?: 'single' | 'complete' | 'average' | 'ward'
  }

  export class HierarchicalClustering {
    constructor(matrix: Matrix, options?: HierarchicalClusteringOptions)
    get_clusters(k: number): number[]
  }
}

# ManifoldDB Logical Plan Specification

This document specifies the unified Intermediate Representation (IR) that both SQL and Cypher compile to. The logical plan is the central abstraction that enables optimization and execution of queries from multiple query languages.

## Design Principles

1. **Language Agnostic** - The IR represents query semantics, not syntax
2. **Complete** - Every SQL and Cypher construct has a logical representation
3. **Optimizable** - Structure enables standard optimization passes
4. **Type-Safe** - Every node has well-defined input/output schemas
5. **Extensible** - New operators can be added for future features

---

## Node Categories

### Category 1: Leaf Nodes (Zero Inputs)

Leaf nodes are data sources that don't consume other plan nodes.

```
┌─────────────────────────────────────────────────────────────────┐
│ Leaf Nodes                                                       │
├─────────────────────────────────────────────────────────────────┤
│ Scan          - Read from a table                               │
│ Values        - Inline literal rows                             │
│ Empty         - Zero-row relation with schema                   │
│ Parameter     - External parameter value                        │
│ CTERef        - Reference to a CTE defined in WITH clause       │
└─────────────────────────────────────────────────────────────────┘
```

### Category 2: Unary Nodes (Single Input)

Unary nodes transform a single input relation.

```
┌─────────────────────────────────────────────────────────────────┐
│ Unary Nodes                                                      │
├─────────────────────────────────────────────────────────────────┤
│ Filter        - Apply predicate (WHERE)                         │
│ Project       - Select/compute columns (SELECT list)            │
│ Aggregate     - Group and aggregate (GROUP BY)                  │
│ Window        - Window function evaluation (OVER)               │
│ Sort          - Order rows (ORDER BY)                           │
│ Limit         - Limit/offset results (LIMIT, OFFSET)            │
│ Distinct      - Remove duplicates (DISTINCT)                    │
│ Alias         - Rename relation                                 │
│ Expand        - Single-hop graph traversal                      │
│ PathScan      - Multi-hop path pattern                          │
│ Unwind        - Expand list to rows (UNWIND)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Category 3: Binary Nodes (Two Inputs)

Binary nodes combine two input relations.

```
┌─────────────────────────────────────────────────────────────────┐
│ Binary Nodes                                                     │
├─────────────────────────────────────────────────────────────────┤
│ Join          - Combine relations (INNER, LEFT, RIGHT, FULL,    │
│                 CROSS, SEMI, ANTI)                              │
│ SetOp         - Set operations (UNION, INTERSECT, EXCEPT)       │
│ Apply         - Correlated subquery evaluation                  │
└─────────────────────────────────────────────────────────────────┘
```

### Category 4: N-ary Nodes (Multiple Inputs)

N-ary nodes can have arbitrary numbers of inputs.

```
┌─────────────────────────────────────────────────────────────────┐
│ N-ary Nodes                                                      │
├─────────────────────────────────────────────────────────────────┤
│ Union         - Multi-way union                                 │
│ Append        - Concatenate relations                           │
└─────────────────────────────────────────────────────────────────┘
```

### Category 5: Graph Mutation Nodes

These nodes modify graph structure (from Cypher writing clauses).

```
┌─────────────────────────────────────────────────────────────────┐
│ Graph Mutation Nodes                                             │
├─────────────────────────────────────────────────────────────────┤
│ GraphCreate   - Create nodes/edges (CREATE)                     │
│ GraphMerge    - Upsert nodes/edges (MERGE)                      │
│ GraphSet      - Set properties/labels (SET)                     │
│ GraphRemove   - Remove properties/labels (REMOVE)               │
│ GraphDelete   - Delete nodes/edges (DELETE, DETACH DELETE)      │
│ Foreach       - Iterate and mutate (FOREACH)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Category 6: Relational Mutation Nodes

These nodes modify table data (from SQL DML).

```
┌─────────────────────────────────────────────────────────────────┐
│ Relational Mutation Nodes                                        │
├─────────────────────────────────────────────────────────────────┤
│ Insert        - Insert rows (INSERT)                            │
│ Update        - Update rows (UPDATE)                            │
│ Delete        - Delete rows (DELETE)                            │
│ Merge         - Upsert rows (SQL MERGE, ON CONFLICT)            │
└─────────────────────────────────────────────────────────────────┘
```

### Category 7: DDL Nodes

These nodes modify schema.

```
┌─────────────────────────────────────────────────────────────────┐
│ DDL Nodes                                                        │
├─────────────────────────────────────────────────────────────────┤
│ CreateTable   - Create table schema                             │
│ AlterTable    - Modify table schema                             │
│ DropTable     - Drop table                                      │
│ CreateIndex   - Create index                                    │
│ DropIndex     - Drop index                                      │
│ CreateView    - Create view                                     │
│ DropView      - Drop view                                       │
│ CreateCollection - Create vector collection                     │
│ DropCollection   - Drop vector collection                       │
└─────────────────────────────────────────────────────────────────┘
```

### Category 8: Vector Search Nodes

These nodes perform vector similarity operations.

```
┌─────────────────────────────────────────────────────────────────┐
│ Vector Search Nodes                                              │
├─────────────────────────────────────────────────────────────────┤
│ AnnSearch     - Approximate nearest neighbor search             │
│ VectorDistance - Compute distance between vectors               │
│ HybridSearch  - Multi-component vector search                   │
└─────────────────────────────────────────────────────────────────┘
```

### Category 9: Procedure Nodes

These nodes invoke stored procedures and algorithms.

```
┌─────────────────────────────────────────────────────────────────┐
│ Procedure Nodes                                                  │
├─────────────────────────────────────────────────────────────────┤
│ ProcedureCall - Invoke procedure (CALL ... YIELD)               │
│ ShortestPath  - Shortest path computation                       │
│ AllShortestPaths - All shortest paths                           │
│ PathAlgorithm - Generic path algorithm                          │
└─────────────────────────────────────────────────────────────────┘
```

### Category 10: Control Flow Nodes

These nodes control query execution flow.

```
┌─────────────────────────────────────────────────────────────────┐
│ Control Flow Nodes                                               │
├─────────────────────────────────────────────────────────────────┤
│ CTE           - Define Common Table Expression                  │
│ RecursiveCTE  - Recursive CTE with working table                │
│ Subquery      - Scalar or lateral subquery                      │
│ Transaction   - Transaction boundary (BEGIN, COMMIT, ROLLBACK)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Node Specifications

### Scan

Reads rows from a table or collection.

```rust
pub struct Scan {
    /// Table name (qualified or simple)
    pub table: TableRef,
    /// Optional alias for the scan
    pub alias: Option<Identifier>,
    /// Columns to project (None = all)
    pub columns: Option<Vec<Identifier>>,
    /// Filter predicates pushed down to scan
    pub filter: Option<Expr>,
    /// Index hint (if any)
    pub index_hint: Option<IndexHint>,
}
```

**SQL Source:** `SELECT * FROM table`
**Cypher Source:** None directly (graph uses Expand)

---

### Values

Represents inline literal values.

```rust
pub struct Values {
    /// Column names for the values
    pub columns: Vec<Identifier>,
    /// Row data as expressions
    pub rows: Vec<Vec<Expr>>,
}
```

**SQL Source:** `VALUES (1, 'a'), (2, 'b')`
**Cypher Source:** `UNWIND [{a: 1}, {a: 2}] AS row`

---

### Filter

Applies a predicate to filter rows.

```rust
pub struct Filter {
    /// The input relation
    pub input: Box<LogicalPlan>,
    /// The filter predicate
    pub predicate: Expr,
}
```

**SQL Source:** `WHERE condition`
**Cypher Source:** `WHERE condition`

---

### Project

Computes a new set of columns from the input.

```rust
pub struct Project {
    /// The input relation
    pub input: Box<LogicalPlan>,
    /// Projection expressions with aliases
    pub projections: Vec<ProjectItem>,
}

pub struct ProjectItem {
    /// The expression to compute
    pub expr: Expr,
    /// Output column name
    pub alias: Identifier,
}
```

**SQL Source:** `SELECT expr AS alias, ...`
**Cypher Source:** `RETURN expr AS alias, ...`

---

### Aggregate

Groups rows and computes aggregate functions.

```rust
pub struct Aggregate {
    /// The input relation
    pub input: Box<LogicalPlan>,
    /// Grouping expressions (GROUP BY)
    pub group_by: Vec<Expr>,
    /// Aggregate expressions
    pub aggregates: Vec<AggregateExpr>,
    /// HAVING filter (applied after aggregation)
    pub having: Option<Expr>,
}

pub struct AggregateExpr {
    /// Aggregate function (COUNT, SUM, etc.)
    pub function: AggregateFunction,
    /// Input expression to the aggregate
    pub args: Vec<Expr>,
    /// Whether DISTINCT is specified
    pub distinct: bool,
    /// Optional FILTER clause
    pub filter: Option<Expr>,
    /// Output column name
    pub alias: Identifier,
}
```

**SQL Source:** `SELECT agg(x) FROM t GROUP BY col HAVING pred`
**Cypher Source:** `RETURN col, count(x)` (implicit grouping)

---

### Window

Evaluates window functions over partitions.

```rust
pub struct Window {
    /// The input relation
    pub input: Box<LogicalPlan>,
    /// Window function specifications
    pub windows: Vec<WindowSpec>,
}

pub struct WindowSpec {
    /// The window function
    pub function: WindowFunction,
    /// Function arguments
    pub args: Vec<Expr>,
    /// Partition expressions
    pub partition_by: Vec<Expr>,
    /// Order within partition
    pub order_by: Vec<OrderByExpr>,
    /// Frame specification
    pub frame: Option<WindowFrame>,
    /// Output column name
    pub alias: Identifier,
}

pub struct WindowFrame {
    /// Frame type: ROWS, RANGE, or GROUPS
    pub frame_type: FrameType,
    /// Start boundary
    pub start: FrameBound,
    /// End boundary
    pub end: FrameBound,
    /// Exclusion mode
    pub exclusion: FrameExclusion,
}

pub enum FrameBound {
    UnboundedPreceding,
    Preceding(Expr),
    CurrentRow,
    Following(Expr),
    UnboundedFollowing,
}

pub enum FrameExclusion {
    NoOthers,
    CurrentRow,
    Group,
    Ties,
}
```

**SQL Source:** `SELECT row_number() OVER (PARTITION BY x ORDER BY y)`
**Cypher Source:** Not directly supported (use procedures)

---

### Sort

Orders rows by expressions.

```rust
pub struct Sort {
    /// The input relation
    pub input: Box<LogicalPlan>,
    /// Sort expressions
    pub order_by: Vec<OrderByExpr>,
}

pub struct OrderByExpr {
    /// Expression to sort by
    pub expr: Expr,
    /// Sort direction
    pub direction: SortDirection,
    /// Null ordering
    pub nulls: NullOrdering,
}

pub enum SortDirection {
    Ascending,
    Descending,
}

pub enum NullOrdering {
    NullsFirst,
    NullsLast,
}
```

**SQL Source:** `ORDER BY expr ASC NULLS FIRST`
**Cypher Source:** `ORDER BY expr ASC`

---

### Limit

Limits and offsets result rows.

```rust
pub struct Limit {
    /// The input relation
    pub input: Box<LogicalPlan>,
    /// Maximum rows to return
    pub limit: Option<Expr>,
    /// Rows to skip
    pub offset: Option<Expr>,
    /// Include ties (WITH TIES)
    pub with_ties: bool,
}
```

**SQL Source:** `LIMIT n OFFSET m` or `FETCH FIRST n ROWS WITH TIES`
**Cypher Source:** `SKIP m LIMIT n`

---

### Distinct

Removes duplicate rows.

```rust
pub struct Distinct {
    /// The input relation
    pub input: Box<LogicalPlan>,
    /// Optional DISTINCT ON expressions (PostgreSQL)
    pub on: Option<Vec<Expr>>,
}
```

**SQL Source:** `SELECT DISTINCT` or `SELECT DISTINCT ON (x, y)`
**Cypher Source:** `RETURN DISTINCT x`

---

### Join

Combines two relations.

```rust
pub struct Join {
    /// Left input
    pub left: Box<LogicalPlan>,
    /// Right input
    pub right: Box<LogicalPlan>,
    /// Join type
    pub join_type: JoinType,
    /// Join condition
    pub condition: JoinCondition,
}

pub enum JoinType {
    Inner,
    LeftOuter,
    RightOuter,
    FullOuter,
    Cross,
    Semi,          // Exists correlation
    Anti,          // Not exists correlation
    LeftMark,      // Mark join for subquery
}

pub enum JoinCondition {
    /// ON clause with predicate
    On(Expr),
    /// USING clause with column list
    Using(Vec<Identifier>),
    /// NATURAL join (implicit column matching)
    Natural,
    /// CROSS join (no condition)
    None,
}
```

**SQL Source:** `FROM a JOIN b ON cond` / `LEFT JOIN` / `CROSS JOIN`
**Cypher Source:** Implicit from pattern matching

---

### SetOp

Performs set operations between two relations.

```rust
pub struct SetOp {
    /// Left input
    pub left: Box<LogicalPlan>,
    /// Right input
    pub right: Box<LogicalPlan>,
    /// Set operation type
    pub op: SetOpType,
    /// Whether to keep duplicates (ALL)
    pub all: bool,
}

pub enum SetOpType {
    Union,
    Intersect,
    Except,
}
```

**SQL Source:** `SELECT ... UNION ALL SELECT ...`
**Cypher Source:** `MATCH ... UNION MATCH ...`

---

### Expand

Single-hop graph traversal (edge expansion).

```rust
pub struct Expand {
    /// Input relation (nodes to expand from)
    pub input: Box<LogicalPlan>,
    /// Source node variable
    pub source: Identifier,
    /// Destination node variable (bound)
    pub destination: Identifier,
    /// Edge variable (optional)
    pub edge: Option<Identifier>,
    /// Traversal direction
    pub direction: Direction,
    /// Edge type filter
    pub edge_types: Option<Vec<Identifier>>,
    /// Edge property filter
    pub edge_filter: Option<Expr>,
    /// Destination node label filter
    pub node_labels: Option<Vec<Identifier>>,
    /// Destination node property filter
    pub node_filter: Option<Expr>,
}

pub enum Direction {
    Outgoing,   // (a)-[]->(b)
    Incoming,   // (a)<-[]-(b)
    Both,       // (a)-[]-(b)
}
```

**SQL Source:** Via MATCH clause extension
**Cypher Source:** `MATCH (a)-[r:TYPE]->(b)`

---

### PathScan

Multi-hop path pattern matching.

```rust
pub struct PathScan {
    /// Input relation (start nodes)
    pub input: Box<LogicalPlan>,
    /// Start node variable
    pub start: Identifier,
    /// End node variable
    pub end: Identifier,
    /// Path variable (optional)
    pub path: Option<Identifier>,
    /// Edge variable (list)
    pub edges: Option<Identifier>,
    /// Nodes variable (list)
    pub nodes: Option<Identifier>,
    /// Path pattern specification
    pub pattern: PathPattern,
    /// Direction
    pub direction: Direction,
}

pub struct PathPattern {
    /// Edge types (None = any)
    pub edge_types: Option<Vec<Identifier>>,
    /// Minimum hops
    pub min_hops: u32,
    /// Maximum hops (None = unbounded)
    pub max_hops: Option<u32>,
    /// Edge filter (applied to each edge)
    pub edge_filter: Option<Expr>,
    /// Node filter (applied to each intermediate node)
    pub node_filter: Option<Expr>,
}
```

**SQL Source:** Via MATCH clause with `*` length
**Cypher Source:** `MATCH p = (a)-[*1..5]->(b)`

---

### Unwind

Expands a list into rows.

```rust
pub struct Unwind {
    /// Input relation
    pub input: Box<LogicalPlan>,
    /// List expression to unwind
    pub list_expr: Expr,
    /// Variable for each element
    pub alias: Identifier,
}
```

**SQL Source:** `unnest(array)` as a table function
**Cypher Source:** `UNWIND list AS item`

---

### GraphCreate

Creates nodes and edges.

```rust
pub struct GraphCreate {
    /// Input relation (provides data for creation)
    pub input: Box<LogicalPlan>,
    /// Patterns to create
    pub patterns: Vec<CreatePattern>,
}

pub struct CreatePattern {
    /// Variable name
    pub variable: Option<Identifier>,
    /// Pattern type
    pub pattern_type: CreatePatternType,
}

pub enum CreatePatternType {
    Node {
        labels: Vec<Identifier>,
        properties: Vec<(Identifier, Expr)>,
    },
    Edge {
        start: Identifier,
        end: Identifier,
        edge_type: Identifier,
        properties: Vec<(Identifier, Expr)>,
    },
}
```

**SQL Source:** Not applicable
**Cypher Source:** `CREATE (n:Label {prop: value})`

---

### GraphMerge

Upserts nodes and edges with conditional logic.

```rust
pub struct GraphMerge {
    /// Input relation
    pub input: Box<LogicalPlan>,
    /// Pattern to merge
    pub pattern: MergePattern,
    /// Actions on create
    pub on_create: Vec<SetAction>,
    /// Actions on match
    pub on_match: Vec<SetAction>,
}

pub struct MergePattern {
    /// Variable for the merged entity
    pub variable: Identifier,
    /// Type of pattern
    pub pattern_type: MergePatternType,
}

pub enum MergePatternType {
    Node {
        labels: Vec<Identifier>,
        match_properties: Vec<(Identifier, Expr)>,
    },
    Edge {
        start: Identifier,
        end: Identifier,
        edge_type: Identifier,
        match_properties: Vec<(Identifier, Expr)>,
    },
}
```

**SQL Source:** `INSERT ... ON CONFLICT` (similar semantics)
**Cypher Source:** `MERGE (n:Label {id: 1}) ON CREATE SET n.x = 1`

---

### GraphSet

Sets properties or labels.

```rust
pub struct GraphSet {
    /// Input relation
    pub input: Box<LogicalPlan>,
    /// Set actions
    pub actions: Vec<SetAction>,
}

pub enum SetAction {
    /// Set a single property
    Property {
        variable: Identifier,
        property: Identifier,
        value: Expr,
    },
    /// Set multiple properties from map
    Properties {
        variable: Identifier,
        map: Expr,
        /// True = replace all, False = merge
        replace: bool,
    },
    /// Add a label
    Label {
        variable: Identifier,
        label: Identifier,
    },
}
```

**SQL Source:** Not applicable
**Cypher Source:** `SET n.prop = value` / `SET n:Label`

---

### GraphRemove

Removes properties or labels.

```rust
pub struct GraphRemove {
    /// Input relation
    pub input: Box<LogicalPlan>,
    /// Remove actions
    pub actions: Vec<RemoveAction>,
}

pub enum RemoveAction {
    Property {
        variable: Identifier,
        property: Identifier,
    },
    Label {
        variable: Identifier,
        label: Identifier,
    },
}
```

**SQL Source:** Not applicable
**Cypher Source:** `REMOVE n.prop` / `REMOVE n:Label`

---

### GraphDelete

Deletes nodes and edges.

```rust
pub struct GraphDelete {
    /// Input relation
    pub input: Box<LogicalPlan>,
    /// Variables to delete
    pub variables: Vec<Identifier>,
    /// Whether to detach (remove relationships first)
    pub detach: bool,
}
```

**SQL Source:** Not applicable
**Cypher Source:** `DELETE n` / `DETACH DELETE n`

---

### AnnSearch

Approximate nearest neighbor vector search.

```rust
pub struct AnnSearch {
    /// Collection or table to search
    pub source: TableRef,
    /// Query vector expression
    pub query_vector: Expr,
    /// Vector column/property name
    pub vector_column: Identifier,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Number of results (k)
    pub k: Expr,
    /// Optional filter
    pub filter: Option<Expr>,
    /// Search parameters
    pub params: AnnSearchParams,
}

pub enum DistanceMetric {
    Euclidean,   // L2
    Cosine,
    InnerProduct,
    Manhattan,   // L1
    Hamming,
}

pub struct AnnSearchParams {
    pub ef_search: Option<u32>,   // HNSW parameter
    pub n_probe: Option<u32>,     // IVF parameter
}
```

**SQL Source:** `ORDER BY embedding <-> $query LIMIT k`
**Cypher Source:** Custom extension

---

### ProcedureCall

Invokes a stored procedure.

```rust
pub struct ProcedureCall {
    /// Procedure name (possibly qualified)
    pub name: QualifiedName,
    /// Arguments
    pub args: Vec<Expr>,
    /// YIELD columns (output projection)
    pub yields: Vec<YieldItem>,
    /// WHERE filter on yields
    pub where_clause: Option<Expr>,
}

pub struct YieldItem {
    /// Result field name
    pub name: Identifier,
    /// Optional alias
    pub alias: Option<Identifier>,
}
```

**SQL Source:** `CALL proc() RETURNING *` (PostgreSQL extension)
**Cypher Source:** `CALL algo.pageRank() YIELD node, score`

---

### RecursiveCTE

Recursive common table expression.

```rust
pub struct RecursiveCTE {
    /// CTE name
    pub name: Identifier,
    /// Column names
    pub columns: Vec<Identifier>,
    /// Initial (non-recursive) query
    pub initial: Box<LogicalPlan>,
    /// Recursive query (references CTE name)
    pub recursive: Box<LogicalPlan>,
    /// Union mode (UNION vs UNION ALL)
    pub union_all: bool,
    /// Search order (depth/breadth first)
    pub search: Option<SearchClause>,
    /// Cycle detection
    pub cycle: Option<CycleClause>,
}

pub struct SearchClause {
    pub mode: SearchMode,
    pub columns: Vec<Identifier>,
    pub result_column: Identifier,
}

pub enum SearchMode {
    DepthFirst,
    BreadthFirst,
}

pub struct CycleClause {
    pub columns: Vec<Identifier>,
    pub cycle_mark: Identifier,
    pub path_column: Option<Identifier>,
}
```

**SQL Source:** `WITH RECURSIVE cte AS (SELECT ... UNION ALL SELECT ... FROM cte)`
**Cypher Source:** Not directly (use path patterns)

---

### Insert

Inserts rows into a table.

```rust
pub struct Insert {
    /// Target table
    pub table: TableRef,
    /// Columns being inserted
    pub columns: Vec<Identifier>,
    /// Source of data
    pub source: InsertSource,
    /// ON CONFLICT handling
    pub on_conflict: Option<OnConflict>,
    /// RETURNING clause
    pub returning: Vec<ProjectItem>,
}

pub enum InsertSource {
    Values(Vec<Vec<Expr>>),
    Query(Box<LogicalPlan>),
    Default,
}

pub struct OnConflict {
    pub columns: Vec<Identifier>,
    pub action: ConflictAction,
}

pub enum ConflictAction {
    DoNothing,
    DoUpdate(Vec<Assignment>),
}
```

**SQL Source:** `INSERT INTO t (cols) VALUES (...) ON CONFLICT DO UPDATE`
**Cypher Source:** Not applicable (use GraphCreate)

---

### Update

Updates rows in a table.

```rust
pub struct Update {
    /// Target table
    pub table: TableRef,
    /// Assignments
    pub assignments: Vec<Assignment>,
    /// FROM clause for joins
    pub from: Vec<Box<LogicalPlan>>,
    /// WHERE filter
    pub filter: Option<Expr>,
    /// RETURNING clause
    pub returning: Vec<ProjectItem>,
}

pub struct Assignment {
    pub column: Identifier,
    pub value: Expr,
}
```

**SQL Source:** `UPDATE t SET col = val WHERE cond`
**Cypher Source:** Not applicable (use GraphSet)

---

### Delete (Relational)

Deletes rows from a table.

```rust
pub struct RelationalDelete {
    /// Target table
    pub table: TableRef,
    /// USING clause for joins
    pub using: Vec<Box<LogicalPlan>>,
    /// WHERE filter
    pub filter: Option<Expr>,
    /// RETURNING clause
    pub returning: Vec<ProjectItem>,
}
```

**SQL Source:** `DELETE FROM t WHERE cond`
**Cypher Source:** Not applicable (use GraphDelete)

---

## Expression Types

The logical plan uses a unified expression system.

```rust
pub enum Expr {
    // Literals
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Vector(Vec<f32>),
    List(Vec<Expr>),
    Map(Vec<(Identifier, Expr)>),

    // References
    Column(Identifier),
    QualifiedColumn(Identifier, Identifier),
    Parameter(ParameterRef),

    // Operators
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOperator,
        expr: Box<Expr>,
    },

    // Functions
    FunctionCall {
        name: QualifiedName,
        args: Vec<Expr>,
        distinct: bool,
        filter: Option<Box<Expr>>,
        over: Option<WindowSpec>,
    },

    // Special forms
    Case {
        operand: Option<Box<Expr>>,
        when_clauses: Vec<(Expr, Expr)>,
        else_result: Option<Box<Expr>>,
    },
    Cast {
        expr: Box<Expr>,
        target_type: DataType,
    },
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
        negated: bool,
    },
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
        negated: bool,
    },
    InSubquery {
        expr: Box<Expr>,
        subquery: Box<LogicalPlan>,
        negated: bool,
    },
    Exists {
        subquery: Box<LogicalPlan>,
        negated: bool,
    },
    ScalarSubquery {
        subquery: Box<LogicalPlan>,
    },

    // List operations
    ListComprehension {
        variable: Identifier,
        list: Box<Expr>,
        filter: Option<Box<Expr>>,
        map: Box<Expr>,
    },
    Subscript {
        expr: Box<Expr>,
        index: Box<Expr>,
    },
    Slice {
        expr: Box<Expr>,
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
    },

    // Map operations
    MapProjection {
        variable: Identifier,
        entries: Vec<MapProjectionEntry>,
    },
    PropertyAccess {
        expr: Box<Expr>,
        property: Identifier,
    },

    // Pattern expressions (for EXISTS patterns)
    PatternExpression {
        pattern: GraphPattern,
    },
}

pub enum BinaryOperator {
    // Arithmetic
    Plus, Minus, Multiply, Divide, Modulo, Power,
    // Comparison
    Eq, NotEq, Lt, LtEq, Gt, GtEq,
    // Logical
    And, Or, Xor,
    // String
    Like, ILike, SimilarTo, RegexMatch,
    StartsWith, EndsWith, Contains,
    // Array/JSON
    ArrayContains, ArrayContainedBy,
    JsonAccess, JsonAccessText, JsonPathAccess,
    // Vector distance
    EuclideanDistance, CosineDistance, InnerProduct,
}

pub enum UnaryOperator {
    Not,
    Negate,
    BitwiseNot,
    IsNull,
    IsNotNull,
    IsTrue,
    IsFalse,
    IsUnknown,
}
```

---

## Type System

Every logical plan node has an associated schema.

```rust
pub struct Schema {
    pub columns: Vec<Column>,
}

pub struct Column {
    pub name: Identifier,
    pub data_type: DataType,
    pub nullable: bool,
}

pub enum DataType {
    // Numeric
    Boolean,
    SmallInt,
    Integer,
    BigInt,
    Real,
    DoublePrecision,
    Numeric { precision: Option<u8>, scale: Option<u8> },

    // String
    Text,
    Varchar { length: Option<u32> },
    Char { length: u32 },

    // Binary
    Bytea,

    // Temporal
    Date,
    Time { with_timezone: bool },
    Timestamp { with_timezone: bool },
    Interval,

    // Complex
    Array { element_type: Box<DataType> },
    Json,
    Jsonb,
    Uuid,

    // Vector
    Vector { dimension: u32 },
    SparseVector { max_dimension: Option<u32> },
    MultiVector { dimension: u32 },
    BinaryVector { bits: u32 },

    // Graph
    Node,
    Edge,
    Path,

    // Special
    Any,
    Unknown,
}
```

---

## Plan Transformation Interface

For optimizer passes:

```rust
pub trait LogicalPlanVisitor {
    fn visit_scan(&mut self, scan: &Scan) -> VisitResult;
    fn visit_filter(&mut self, filter: &Filter) -> VisitResult;
    fn visit_project(&mut self, project: &Project) -> VisitResult;
    // ... for each node type
}

pub trait LogicalPlanRewriter {
    fn rewrite_scan(&mut self, scan: Scan) -> LogicalPlan;
    fn rewrite_filter(&mut self, filter: Filter) -> LogicalPlan;
    fn rewrite_project(&mut self, project: Project) -> LogicalPlan;
    // ... for each node type
}
```

---

## References

- [QUERY_IMPLEMENTATION_ROADMAP.md](./QUERY_IMPLEMENTATION_ROADMAP.md) - Implementation phases
- [COVERAGE_MATRICES.md](./COVERAGE_MATRICES.md) - Feature coverage tracking
- [Apache DataFusion](https://datafusion.apache.org/) - Reference logical plan design
- [Calcite](https://calcite.apache.org/) - Relational algebra reference

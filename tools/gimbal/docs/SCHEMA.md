# Gimbal Knowledge Graph Schema

This document defines the graph schema for Gimbal's code and documentation knowledge base. It covers entity types, relationships, properties, and the queries/operations the schema must support.

## Overview

Gimbal ingests code repositories, documentation, and web content into a searchable knowledge graph with vector embeddings. The schema supports:

- Hierarchical containment (directories, files, symbols)
- Multiple content types (code, documentation, web pages)
- Vector similarity search with filtering
- Version tracking and temporal relationships
- Cross-reference relationships between content

---

## Entity Types

### Container Entities

These represent structural containers that hold other content.

#### Domain

Represents a website domain, the root URL for crawled web content.

| Property | Type | Description |
|----------|------|-------------|
| `url` | String | Base URL (e.g., "https://docs.rs", "https://redis.io/docs") |
| `name` | String | Human-readable name |
| `crawled_at` | Timestamp | When the domain was last crawled |

#### Repository

A version-controlled code repository.

| Property | Type | Description |
|----------|------|-------------|
| `url` | String | Remote URL (e.g., GitHub URL) |
| `path` | String | Local filesystem path |
| `name` | String | Repository name |
| `default_branch` | String | Main branch name (e.g., "main") |
| `owner` | String | Owner username or org name |
| `topics` | String[] | Repository topic tags |
| `languages` | Map<String, Float> | Language breakdown by percentage |

#### Directory

A filesystem directory within a repository or local path.

| Property | Type | Description |
|----------|------|-------------|
| `path` | String | Absolute or relative path |
| `name` | String | Directory name |

#### Module

A language-specific module (e.g., Rust module, Python package).

| Property | Type | Description |
|----------|------|-------------|
| `path` | String | Module path |
| `name` | String | Module name |
| `language` | String | Programming language |
| `visibility` | String | Public/private/crate |

---

### Source Entities

These represent content sources - files, pages, documents.

#### CodeFile

A source code file.

| Property | Type | Description |
|----------|------|-------------|
| `path` | String | File path |
| `name` | String | Filename |
| `language` | String | Programming language (rust, python, typescript, etc.) |
| `extension` | String | File extension |
| `size_bytes` | Integer | File size |
| `hash` | String | Content hash for change detection |

#### Document

A documentation file (markdown, MDX, RST, etc.).

| Property | Type | Description |
|----------|------|-------------|
| `path` | String | File path |
| `name` | String | Filename |
| `format` | String | Document format (markdown, mdx, rst, asciidoc) |
| `title` | String | Document title (extracted from content) |

#### Page

Web content scraped from a URL.

| Property | Type | Description |
|----------|------|-------------|
| `url` | String | Source URL |
| `title` | String | Page title |
| `page_type` | String | Classification (documentation, article, discussion, api_reference, tutorial) |
| `fetched_at` | Timestamp | When the page was fetched |

#### Paper

Academic paper or technical report (typically PDF).

| Property | Type | Description |
|----------|------|-------------|
| `path` | String | Local file path |
| `url` | String | Source URL if downloaded |
| `title` | String | Paper title |
| `authors` | String[] | Author names |
| `published_date` | Date | Publication date |
| `doi` | String | DOI identifier if available |
| `arxiv_id` | String | arXiv ID if available |

---

### Content Entities

These represent the actual indexed content with embeddings.

#### Symbol

A code symbol extracted from a source file. This is the primary entity for code search.

| Property | Type | Description |
|----------|------|-------------|
| `name` | String | Symbol name (e.g., "parse_config") |
| `kind` | String | Symbol kind (see Symbol Kinds below) |
| `signature` | String | Full signature (e.g., "pub fn parse_config(path: &Path) -> Result<Config>") |
| `visibility` | String | Visibility level (public, private, crate, super) |
| `language` | String | Programming language |
| `content` | String | Full source text of the symbol |
| `line_start` | Integer | Starting line number (1-indexed) |
| `line_end` | Integer | Ending line number (1-indexed) |
| `byte_start` | Integer | Starting byte offset |
| `byte_end` | Integer | Ending byte offset |

**Symbol Kinds:**
- `function` - Standalone function
- `method` - Method within impl/class
- `struct` - Struct definition
- `enum` - Enum definition
- `trait` - Trait definition
- `interface` - Interface (TypeScript, Go)
- `class` - Class definition
- `impl` - Implementation block
- `module` - Module declaration
- `constant` - Constant definition
- `type` - Type alias
- `field` - Struct/class field
- `macro` - Macro definition
- `comment` - Documentation comment (optional, for TODO tracking etc.)

#### Section

A section of documentation, extracted from a Document or Page.

| Property | Type | Description |
|----------|------|-------------|
| `heading` | String | Section heading text |
| `level` | Integer | Heading level (1-6 for markdown) |
| `content` | String | Section content including sub-content |
| `line_start` | Integer | Starting line number |
| `line_end` | Integer | Ending line number |

---

### Reference Entities

#### Owner

Repository owner (user or organization).

| Property | Type | Description |
|----------|------|-------------|
| `name` | String | Username or org name |
| `type` | String | "user" or "organization" |
| `url` | String | Profile URL |

#### Dependency

A package dependency (parsed from Cargo.toml, package.json, etc.).

| Property | Type | Description |
|----------|------|-------------|
| `name` | String | Package name |
| `version` | String | Version specifier |
| `registry` | String | Package registry (crates.io, npm, pypi) |
| `dev` | Boolean | Whether this is a dev dependency |
| `optional` | Boolean | Whether this is optional |

#### Version

A specific version of a repository or package.

| Property | Type | Description |
|----------|------|-------------|
| `version` | String | Version string (e.g., "1.0.0") |
| `tag` | String | Git tag if applicable |
| `commit` | String | Commit hash |
| `released_at` | Timestamp | Release timestamp |

---

## Edge Types

### Hierarchical Containment

All parent-child relationships use `CONTAINS`, pointing from parent to child.

#### CONTAINS

Represents structural containment. The parent contains the child.

| From | To | Description |
|------|----|-------------|
| Domain | Page | Domain contains crawled pages |
| Repository | Directory | Repository contains top-level directories |
| Repository | CodeFile | Repository contains top-level files |
| Repository | Document | Repository contains top-level docs |
| Directory | Directory | Directory contains subdirectories |
| Directory | CodeFile | Directory contains code files |
| Directory | Document | Directory contains documents |
| CodeFile | Symbol | File contains top-level symbols |
| Symbol | Symbol | Symbol contains nested symbols (e.g., impl contains methods) |
| Document | Section | Document contains sections |
| Section | Section | Section contains subsections |
| Page | Section | Page contains content sections |

**Properties on CONTAINS edge:**
| Property | Type | Description |
|----------|------|-------------|
| `order` | Integer | Ordering index for sequential children |

---

### Sequential Relationships

#### FOLLOWED_BY

Represents sequential ordering within a container.

| From | To | Description |
|------|----|-------------|
| Symbol | Symbol | Next symbol in file order |
| Section | Section | Next section in document order |

---

### Dependency Relationships

#### DEPENDS_ON

Represents a dependency from one package/repository to another.

| From | To | Description |
|------|----|-------------|
| Repository | Dependency | Repository depends on package |
| Dependency | Repository | Dependency resolves to repository (if known) |

**Properties on DEPENDS_ON edge:**
| Property | Type | Description |
|----------|------|-------------|
| `version` | String | Version constraint |
| `dev` | Boolean | Dev dependency flag |
| `optional` | Boolean | Optional dependency flag |

---

### Version Relationships

#### REPLACED_BY

Represents version succession - one version is replaced by a newer one.

| From | To | Description |
|------|----|-------------|
| Version | Version | Older version replaced by newer |
| CodeFile | CodeFile | File version replaced (for local tracking) |
| Symbol | Symbol | Symbol definition changed |

**Properties on REPLACED_BY edge:**
| Property | Type | Description |
|----------|------|-------------|
| `replaced_at` | Timestamp | When the replacement occurred |
| `commit` | String | Commit hash if applicable |

#### FORKED_FROM

Represents repository fork relationship.

| From | To | Description |
|------|----|-------------|
| Repository | Repository | Fork source |

#### BRANCHED_FROM

Represents git branch relationship.

| From | To | Description |
|------|----|-------------|
| Version | Version | Branch point |

---

### Reference Relationships

#### REFERENCES

Generic reference from one content to another.

| From | To | Description |
|------|----|-------------|
| Section | Section | Documentation cross-reference |
| Symbol | Symbol | Code reference (future: imports) |
| Paper | Paper | Citation |

#### LINKS_TO

URL hyperlink between pages.

| From | To | Description |
|------|----|-------------|
| Page | Page | Hyperlink from one page to another |
| Section | Page | Documentation links to external page |

---

### Ownership

#### OWNED_BY

Repository ownership.

| From | To | Description |
|------|----|-------------|
| Repository | Owner | Repository owned by user/org |

---

## Vector Storage

Symbols and Sections are the primary entities with vector embeddings.

### Vector Types

| Name | Type | Description |
|------|------|-------------|
| `dense` | Dense vector | Standard dense embedding (e.g., BGE, Qwen) |
| `sparse` | Sparse vector | SPLADE sparse embedding for keyword matching |
| `colbert` | Multi-vector | ColBERT token embeddings for late interaction |

### Indexed Payload Fields

These fields should be indexed for filtering during vector search:

| Field | Type | Filter Operations |
|-------|------|-------------------|
| `language` | String | Equality (e.g., `language = "rust"`) |
| `kind` | String | Equality (e.g., `kind = "function"`) |
| `visibility` | String | Equality (e.g., `visibility = "public"`) |
| `file_path` | String | Prefix match, contains |
| `name` | String | Prefix match, contains |

---

## Required Query Capabilities

### Graph Traversal

1. **Recursive containment**: Given a Directory or Repository, find all contained entities recursively.
   ```
   MATCH (r:Repository)-[:CONTAINS*]->(s:Symbol)
   WHERE r.name = "gimbal"
   RETURN s
   ```

2. **Parent lookup**: Given a Symbol, find its containing file and directory path.
   ```
   MATCH (s:Symbol)<-[:CONTAINS*]-(f:CodeFile)<-[:CONTAINS*]-(d:Directory)
   WHERE s.name = "parse_config"
   RETURN f, d
   ```

3. **Sibling navigation**: Given a Symbol, find the next/previous symbols in the file.
   ```
   MATCH (s:Symbol)-[:FOLLOWED_BY]->(next:Symbol)
   WHERE s.name = "parse_config"
   RETURN next
   ```

4. **Nested symbol hierarchy**: Given an impl block, find all methods it contains.
   ```
   MATCH (impl:Symbol)-[:CONTAINS]->(method:Symbol)
   WHERE impl.kind = "impl" AND impl.name = "Config"
   RETURN method
   ```

### Vector Search with Filters

1. **Search by language**:
   ```
   VECTOR SEARCH symbols
   WHERE language = "rust"
   QUERY <embedding>
   LIMIT 10
   ```

2. **Search public API only**:
   ```
   VECTOR SEARCH symbols
   WHERE visibility = "public" AND kind IN ["function", "struct", "trait"]
   QUERY <embedding>
   LIMIT 10
   ```

3. **Search within repository**:
   ```
   VECTOR SEARCH symbols
   WHERE file_path STARTS WITH "/path/to/repo"
   QUERY <embedding>
   LIMIT 10
   ```

4. **Search documentation only**:
   ```
   VECTOR SEARCH sections
   QUERY <embedding>
   LIMIT 10
   ```

### Dependency Queries

1. **List all dependencies of a repository**:
   ```
   MATCH (r:Repository)-[:DEPENDS_ON]->(d:Dependency)
   WHERE r.name = "gimbal"
   RETURN d
   ```

2. **Find repositories using a package**:
   ```
   MATCH (r:Repository)-[:DEPENDS_ON]->(d:Dependency)
   WHERE d.name = "tokio"
   RETURN r
   ```

### Version Queries

1. **Version history**: Find all versions of a symbol through REPLACED_BY chain.
   ```
   MATCH (s:Symbol)-[:REPLACED_BY*]->(newer:Symbol)
   WHERE s.name = "parse_config"
   RETURN s, newer
   ```

---

## Implementation Phases

### Phase 1: Core Schema (Current Priority)

**Entities:**
- Directory
- CodeFile
- Document
- Symbol (with all properties)
- Section

**Edges:**
- CONTAINS (with order property)
- FOLLOWED_BY

**Capabilities:**
- Nested symbol extraction and containment
- Vector search with language/kind/visibility filters
- Basic graph traversal (parent lookup, containment)

### Phase 2: Repository Integration

**Entities:**
- Repository
- Owner
- Dependency

**Edges:**
- OWNED_BY
- DEPENDS_ON

**Capabilities:**
- Parse Cargo.toml/package.json for dependencies
- Track repository metadata (topics, languages)

### Phase 3: Web Content

**Entities:**
- Domain
- Page

**Edges:**
- LINKS_TO

**Capabilities:**
- Web crawling and page extraction
- Cross-reference tracking between pages

### Phase 4: Version Tracking

**Entities:**
- Version

**Edges:**
- REPLACED_BY
- FORKED_FROM
- BRANCHED_FROM

**Capabilities:**
- Track changes over time
- Version-specific documentation linking

---

## Notes for ManifoldDB Implementation

1. **Edge traversal direction**: All edges should be traversable in both directions efficiently. We only store one direction (e.g., CONTAINS from parent to child) but queries need to traverse backwards (find parent from child).

2. **Recursive traversal**: The `[:CONTAINS*]` pattern (variable-length path) is essential for walking the hierarchy. This should be efficient for trees up to ~10 levels deep.

3. **Vector search with graph filters**: Need ability to filter vector search results based on graph relationships, e.g., "search symbols contained within this repository."

4. **Payload indexing**: The vector collection payload fields (language, kind, visibility, file_path, name) need to support efficient filtering during ANN search.

5. **Edge properties**: Edges like CONTAINS need to support properties (e.g., `order` for sequencing). These should be queryable.

6. **Upsert semantics**: When re-ingesting content, we need to update existing entities rather than create duplicates. Entity identity should be based on (file_path + byte_range) for symbols, (file_path + heading) for sections.

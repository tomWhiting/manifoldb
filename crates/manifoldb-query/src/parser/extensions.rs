//! Graph and vector syntax extensions.
//!
//! This module provides parsing for custom SQL extensions:
//! - Graph pattern matching (MATCH clause)
//! - Vector distance operators (<->, <=>, <#>, <##>)
//! - Weighted shortest path queries (SHORTEST PATH ... WEIGHTED BY)
//!
//! # Extended Syntax Examples
//!
//! ## Vector Similarity Search
//! ```sql
//! SELECT * FROM docs ORDER BY embedding <-> $query LIMIT 10;
//! ```
//!
//! ## Multi-Vector MaxSim Search (ColBERT-style)
//! ```sql
//! SELECT * FROM docs ORDER BY token_embeddings <##> $query_tokens LIMIT 10;
//! ```
//!
//! ## Graph Pattern Matching
//! ```sql
//! SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1;
//! ```
//!
//! ## Weighted Shortest Path
//! ```sql
//! -- Unweighted shortest path (BFS)
//! SELECT SHORTEST PATH (a)-[*]->(b)
//! WHERE a.id = 1 AND b.id = 10;
//!
//! -- Weighted shortest path (Dijkstra) using edge property
//! SELECT SHORTEST PATH (a)-[:ROAD*]->(b) WEIGHTED BY distance
//! WHERE a.name = 'New York' AND b.name = 'Los Angeles';
//!
//! -- All shortest paths of equal length
//! SELECT ALL SHORTEST PATHS (a)-[*]->(b)
//! WHERE a.id = 1 AND b.id = 10;
//! ```
//!
//! ## Combined Query
//! ```sql
//! SELECT d.*, a.name
//! FROM docs d
//! MATCH (d)-[:AUTHORED_BY]->(a)
//! WHERE embedding <-> $query < 0.5;
//! ```

use crate::ast::{
    BinaryOp, CallStatement, CreateCollectionStatement, CreateGraphStatement, CreateNodeRef,
    CreatePathStep, CreatePattern, DataType, DeleteGraphStatement, DistanceMetric,
    DropCollectionStatement, EdgeDirection, EdgeLength, EdgePattern, Expr, GraphPattern,
    Identifier, MatchStatement, MergeGraphStatement, MergePattern, NodePattern, OrderByExpr,
    ParameterRef, PathPattern, PayloadFieldDef, PropertyCondition, QualifiedName,
    RemoveGraphStatement, RemoveItem, ReturnItem, SelectStatement, SetAction, SetGraphStatement,
    ShortestPathPattern, Statement, VectorDef, VectorTypeDef, WeightSpec, YieldItem,
};
use crate::error::{ParseError, ParseResult};
use crate::parser::sql;

/// Extended SQL parser with graph and vector support.
pub struct ExtendedParser;

impl ExtendedParser {
    /// Parses an extended SQL query with graph and vector syntax.
    ///
    /// This function handles:
    /// 1. Cypher-style standalone MATCH statements (MATCH ... RETURN)
    /// 2. Pre-processing to convert custom operators to function calls
    /// 3. Standard SQL parsing
    /// 4. Post-processing to extract MATCH clauses
    /// 5. Restoration of vector operators from function calls
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is syntactically invalid.
    pub fn parse(input: &str) -> ParseResult<Vec<Statement>> {
        if input.trim().is_empty() {
            return Err(ParseError::EmptyQuery);
        }

        // Check for standalone MATCH statement (Cypher-style)
        if Self::is_standalone_match(input) {
            return Self::parse_standalone_match(input);
        }

        // Check for CREATE COLLECTION statement (not supported by sqlparser)
        if Self::is_create_collection(input) {
            return Self::parse_create_collection(input);
        }

        // Check for DROP COLLECTION statement (not supported by sqlparser)
        if Self::is_drop_collection(input) {
            return Self::parse_drop_collection(input);
        }

        // Check for Cypher-style CREATE graph statement
        if Self::is_cypher_create(input) {
            return Self::parse_cypher_create(input);
        }

        // Check for Cypher-style MERGE graph statement
        if Self::is_cypher_merge(input) {
            return Self::parse_cypher_merge(input);
        }

        // Check for Cypher-style SET statement (MATCH ... SET ...)
        if Self::is_cypher_set(input) {
            return Self::parse_cypher_set(input);
        }

        // Check for Cypher-style DELETE statement (MATCH ... DELETE ...)
        if Self::is_cypher_delete(input) {
            return Self::parse_cypher_delete(input);
        }

        // Check for Cypher-style REMOVE statement (MATCH ... REMOVE ...)
        if Self::is_cypher_remove(input) {
            return Self::parse_cypher_remove(input);
        }

        // Check for CALL with YIELD (requires custom parsing for Cypher-style YIELD clause)
        if Self::is_call_with_yield(input) {
            return Self::parse_call_with_yield(input);
        }

        // Step 1: Extract MATCH and OPTIONAL MATCH clauses (they're not valid SQL)
        let (sql_without_match, match_patterns, optional_patterns) =
            Self::extract_match_clauses(input)?;

        // Step 2: Pre-process vector operators
        let preprocessed = Self::preprocess_vector_ops(&sql_without_match);

        // Step 3: Parse the SQL
        let mut statements = sql::parse_sql(&preprocessed)?;

        // Step 4: Post-process to restore vector operators and add match clauses
        for (i, stmt) in statements.iter_mut().enumerate() {
            Self::restore_vector_ops(stmt);
            if let Some(pattern) = match_patterns.get(i) {
                Self::add_match_clause(stmt, pattern.clone());
            }
            if let Some(opt_patterns) = optional_patterns.get(i) {
                for pattern in opt_patterns {
                    Self::add_optional_match_clause(stmt, pattern.clone());
                }
            }
        }

        Ok(statements)
    }

    /// Checks if the input is a standalone MATCH statement (Cypher-style).
    ///
    /// A standalone MATCH statement starts with MATCH and contains RETURN.
    fn is_standalone_match(input: &str) -> bool {
        let trimmed = input.trim();
        let upper = trimmed.to_uppercase();

        // Must start with MATCH (not SELECT ... MATCH)
        if !upper.starts_with("MATCH") {
            return false;
        }

        // Must contain RETURN keyword
        if !upper.contains("RETURN") {
            return false;
        }

        // Must NOT contain Cypher DML keywords (those are handled separately)
        // Check for SET, DELETE, REMOVE which indicate mutation operations
        if upper.contains(" SET ") && !upper.contains("MERGE") {
            return false; // This is a SET statement
        }
        if upper.contains(" DELETE ") || upper.contains("DETACH DELETE") {
            return false; // This is a DELETE statement
        }
        if upper.contains(" REMOVE ") {
            return false; // This is a REMOVE statement
        }

        true
    }

    /// Checks if the input is a CREATE COLLECTION statement.
    fn is_create_collection(input: &str) -> bool {
        let upper = input.trim().to_uppercase();
        upper.starts_with("CREATE COLLECTION")
            || upper.starts_with("CREATE IF NOT EXISTS COLLECTION")
    }

    /// Checks if the input is a DROP COLLECTION statement.
    fn is_drop_collection(input: &str) -> bool {
        let upper = input.trim().to_uppercase();
        upper.starts_with("DROP COLLECTION") || upper.starts_with("DROP IF EXISTS COLLECTION")
    }

    /// Checks if the input is a Cypher-style CREATE statement for graphs.
    ///
    /// A Cypher CREATE starts with either:
    /// - `CREATE (...)` - create node(s)
    /// - `MATCH ... CREATE ...` - create after matching
    ///
    /// It should NOT be confused with `CREATE TABLE` or `CREATE INDEX`.
    fn is_cypher_create(input: &str) -> bool {
        let trimmed = input.trim();
        let upper = trimmed.to_uppercase();

        // Check for CREATE followed by graph pattern
        if upper.starts_with("CREATE") {
            let after_create = trimmed[6..].trim_start();
            // Must be followed by '(' for a node pattern
            // NOT followed by TABLE, INDEX, COLLECTION, etc.
            if after_create.starts_with('(') {
                return true;
            }
        }

        // Check for MATCH ... CREATE pattern
        if upper.starts_with("MATCH") && upper.contains("CREATE") {
            // Ensure CREATE is followed by a graph pattern, not TABLE/INDEX
            if let Some(pos) = upper.find("CREATE") {
                let after_create = &trimmed[pos + 6..].trim_start();
                if after_create.starts_with('(') {
                    return true;
                }
            }
        }

        false
    }

    /// Checks if the input is a Cypher-style MERGE statement.
    fn is_cypher_merge(input: &str) -> bool {
        let trimmed = input.trim();
        let upper = trimmed.to_uppercase();

        // MERGE starts the statement or follows MATCH
        if upper.starts_with("MERGE") {
            let after_merge = trimmed[5..].trim_start();
            // Must be followed by '(' for a node pattern
            return after_merge.starts_with('(');
        }

        // Check for MATCH ... MERGE pattern
        if upper.starts_with("MATCH") && upper.contains("MERGE") {
            if let Some(pos) = upper.find("MERGE") {
                let after_merge = &trimmed[pos + 5..].trim_start();
                if after_merge.starts_with('(') {
                    return true;
                }
            }
        }

        false
    }

    /// Checks if the input is a Cypher-style SET statement.
    ///
    /// A Cypher SET must follow a MATCH clause (standalone SET is not valid Cypher).
    fn is_cypher_set(input: &str) -> bool {
        let upper = input.trim().to_uppercase();

        // Must start with MATCH and contain SET (but not ON MATCH SET from MERGE)
        if upper.starts_with("MATCH") && upper.contains(" SET ") {
            // Ensure it's not a MERGE statement with ON MATCH SET
            !upper.contains("MERGE")
        } else {
            false
        }
    }

    /// Checks if the input is a Cypher-style DELETE statement.
    ///
    /// DELETE follows MATCH. DETACH DELETE is also supported.
    fn is_cypher_delete(input: &str) -> bool {
        let upper = input.trim().to_uppercase();

        // Must start with MATCH and contain DELETE
        if upper.starts_with("MATCH") {
            upper.contains(" DELETE ") || upper.contains("DETACH DELETE")
        } else {
            false
        }
    }

    /// Checks if the input is a Cypher-style REMOVE statement.
    ///
    /// REMOVE follows MATCH to remove properties or labels.
    fn is_cypher_remove(input: &str) -> bool {
        let upper = input.trim().to_uppercase();

        // Must start with MATCH and contain REMOVE
        upper.starts_with("MATCH") && upper.contains(" REMOVE ")
    }

    /// Parses a DROP COLLECTION statement.
    ///
    /// Syntax:
    /// ```text
    /// DROP COLLECTION [IF EXISTS] name [, name2, ...];
    /// ```
    fn parse_drop_collection(input: &str) -> ParseResult<Vec<Statement>> {
        let input = input.trim().trim_end_matches(';');
        let upper = input.to_uppercase();

        // Parse IF EXISTS
        let (if_exists, after_if_exists) = if upper.starts_with("DROP IF EXISTS") {
            (true, input[14..].trim_start()) // "DROP IF EXISTS" = 14 chars
        } else if upper.starts_with("DROP") {
            (false, input[4..].trim_start()) // "DROP" = 4 chars
        } else {
            return Err(ParseError::SqlSyntax("expected DROP keyword".to_string()));
        };

        // Parse COLLECTION keyword
        let upper_rest = after_if_exists.to_uppercase();
        if !upper_rest.starts_with("COLLECTION") {
            return Err(ParseError::SqlSyntax(
                "expected COLLECTION keyword after DROP".to_string(),
            ));
        }
        let after_collection = after_if_exists[10..].trim_start(); // "COLLECTION" = 10 chars

        // Parse collection names (comma-separated)
        let names: Vec<Identifier> =
            after_collection.split(',').map(|s| Identifier::new(s.trim())).collect();

        if names.is_empty() || names.iter().any(|n| n.name.is_empty()) {
            return Err(ParseError::SqlSyntax("expected collection name(s)".to_string()));
        }

        let mut stmt = DropCollectionStatement::new(names);
        if if_exists {
            stmt = stmt.if_exists();
        }

        Ok(vec![Statement::DropCollection(stmt)])
    }

    /// Checks if the input is a CALL statement with YIELD clause.
    fn is_call_with_yield(input: &str) -> bool {
        let upper = input.trim().to_uppercase();
        upper.starts_with("CALL") && upper.contains("YIELD")
    }

    /// Parses a CALL statement with YIELD clause.
    ///
    /// Syntax:
    /// ```text
    /// CALL procedure.name(arg1, arg2, ...) YIELD col1, col2 AS alias, ... [WHERE condition]
    /// CALL procedure.name(arg1, arg2, ...) YIELD *
    /// ```
    fn parse_call_with_yield(input: &str) -> ParseResult<Vec<Statement>> {
        let input = input.trim().trim_end_matches(';');

        // Split into CALL part and YIELD part
        let upper = input.to_uppercase();
        let yield_pos = upper
            .find(" YIELD ")
            .ok_or_else(|| ParseError::SqlSyntax("expected YIELD clause after CALL".to_string()))?;

        let call_part = &input[4..yield_pos].trim(); // Skip "CALL "
        let yield_part = &input[yield_pos + 7..]; // Skip " YIELD "

        // Parse procedure name and arguments
        let (procedure_name, arguments) = Self::parse_procedure_call(call_part)?;

        // Parse YIELD clause and optional WHERE
        let (yield_items, where_clause) = Self::parse_yield_clause(yield_part)?;

        let mut stmt = CallStatement::new(procedure_name, arguments).yield_items(yield_items);

        if let Some(condition) = where_clause {
            stmt = stmt.where_clause(condition);
        }

        Ok(vec![Statement::Call(Box::new(stmt))])
    }

    /// Parses a procedure call (name and arguments).
    fn parse_procedure_call(input: &str) -> ParseResult<(QualifiedName, Vec<Expr>)> {
        // Find the opening paren
        let paren_pos = input.find('(').ok_or_else(|| {
            ParseError::SqlSyntax("expected '(' after procedure name".to_string())
        })?;

        let name_part = input[..paren_pos].trim();
        let args_part = input[paren_pos + 1..].trim();

        // Parse procedure name (can be qualified like algo.pageRank)
        let procedure_name = Self::parse_qualified_name(name_part)?;

        // Find closing paren
        let close_pos = args_part.rfind(')').ok_or_else(|| {
            ParseError::SqlSyntax("expected ')' to close procedure arguments".to_string())
        })?;

        let args_str = args_part[..close_pos].trim();

        // Parse arguments
        let arguments =
            if args_str.is_empty() { vec![] } else { Self::parse_argument_list(args_str)? };

        Ok((procedure_name, arguments))
    }

    /// Parses a qualified name (e.g., "algo.pageRank").
    fn parse_qualified_name(input: &str) -> ParseResult<QualifiedName> {
        let parts: Vec<&str> = input.split('.').collect();
        if parts.is_empty() || parts.iter().any(|p| p.trim().is_empty()) {
            return Err(ParseError::SqlSyntax("invalid procedure name".to_string()));
        }

        let identifiers: Vec<Identifier> =
            parts.iter().map(|s| Identifier::new(s.trim())).collect();
        Ok(QualifiedName::new(identifiers))
    }

    /// Parses a comma-separated argument list.
    fn parse_argument_list(input: &str) -> ParseResult<Vec<Expr>> {
        // Handle nested parentheses and quotes for proper splitting
        let mut args = Vec::new();
        let mut current = String::new();
        let mut depth = 0;
        let mut in_string = false;
        let mut string_char = '"';

        for ch in input.chars() {
            match ch {
                '"' | '\'' if !in_string => {
                    in_string = true;
                    string_char = ch;
                    current.push(ch);
                }
                c if c == string_char && in_string => {
                    in_string = false;
                    current.push(ch);
                }
                '(' if !in_string => {
                    depth += 1;
                    current.push(ch);
                }
                ')' if !in_string => {
                    depth -= 1;
                    current.push(ch);
                }
                ',' if !in_string && depth == 0 => {
                    args.push(Self::parse_simple_expr(current.trim())?);
                    current = String::new();
                }
                _ => current.push(ch),
            }
        }

        // Don't forget the last argument
        if !current.trim().is_empty() {
            args.push(Self::parse_simple_expr(current.trim())?);
        }

        Ok(args)
    }

    /// Parses a simple expression (literal, identifier, or parameter).
    fn parse_simple_expr(input: &str) -> ParseResult<Expr> {
        let trimmed = input.trim();

        // NULL
        if trimmed.eq_ignore_ascii_case("null") {
            return Ok(Expr::null());
        }

        // Boolean
        if trimmed.eq_ignore_ascii_case("true") {
            return Ok(Expr::boolean(true));
        }
        if trimmed.eq_ignore_ascii_case("false") {
            return Ok(Expr::boolean(false));
        }

        // String literal
        if (trimmed.starts_with('\'') && trimmed.ends_with('\''))
            || (trimmed.starts_with('"') && trimmed.ends_with('"'))
        {
            let s = &trimmed[1..trimmed.len() - 1];
            return Ok(Expr::string(s));
        }

        // Parameter ($name or $1)
        if let Some(rest) = trimmed.strip_prefix('$') {
            if let Ok(pos) = rest.parse::<u32>() {
                return Ok(Expr::Parameter(ParameterRef::Positional(pos)));
            }
            return Ok(Expr::Parameter(ParameterRef::Named(rest.to_string())));
        }

        // Integer
        if let Ok(i) = trimmed.parse::<i64>() {
            return Ok(Expr::integer(i));
        }

        // Float
        if let Ok(f) = trimmed.parse::<f64>() {
            return Ok(Expr::float(f));
        }

        // Column reference (identifier or qualified)
        Ok(Expr::Column(Self::parse_qualified_name(trimmed)?))
    }

    /// Parses a YIELD clause including optional WHERE.
    fn parse_yield_clause(input: &str) -> ParseResult<(Vec<YieldItem>, Option<Expr>)> {
        let upper = input.to_uppercase();

        // Check for WHERE clause
        let (yield_part, where_clause) = if let Some(where_pos) = upper.find(" WHERE ") {
            let yield_str = input[..where_pos].trim();
            let where_str = input[where_pos + 7..].trim();
            let condition = Self::parse_where_condition(where_str)?;
            (yield_str, Some(condition))
        } else {
            (input.trim(), None)
        };

        // Parse YIELD items
        let yield_items = Self::parse_yield_items(yield_part)?;

        Ok((yield_items, where_clause))
    }

    /// Parses YIELD items (column names with optional aliases).
    fn parse_yield_items(input: &str) -> ParseResult<Vec<YieldItem>> {
        let trimmed = input.trim();

        // Handle YIELD *
        if trimmed == "*" {
            return Ok(vec![YieldItem::Wildcard]);
        }

        // Parse comma-separated items
        let mut items = Vec::new();
        for part in trimmed.split(',') {
            let part = part.trim();
            let upper_part = part.to_uppercase();

            // Check for AS alias
            if let Some(as_pos) = upper_part.find(" AS ") {
                let name = part[..as_pos].trim();
                let alias = part[as_pos + 4..].trim();
                items.push(YieldItem::aliased(name, alias));
            } else {
                items.push(YieldItem::column(part));
            }
        }

        if items.is_empty() {
            return Err(ParseError::SqlSyntax("expected at least one YIELD item".to_string()));
        }

        Ok(items)
    }

    /// Parses a WHERE condition after YIELD.
    fn parse_where_condition(input: &str) -> ParseResult<Expr> {
        // For now, handle simple binary comparisons
        // Format: column_name op value
        let trimmed = input.trim();

        // Try to find a comparison operator
        for (op_str, op) in &[
            (">=", BinaryOp::GtEq),
            ("<=", BinaryOp::LtEq),
            ("<>", BinaryOp::NotEq),
            ("!=", BinaryOp::NotEq),
            ("=", BinaryOp::Eq),
            (">", BinaryOp::Gt),
            ("<", BinaryOp::Lt),
        ] {
            if let Some(pos) = trimmed.find(op_str) {
                let left = trimmed[..pos].trim();
                let right = trimmed[pos + op_str.len()..].trim();

                let left_expr = Self::parse_simple_expr(left)?;
                let right_expr = Self::parse_simple_expr(right)?;

                return Ok(Expr::BinaryOp {
                    left: Box::new(left_expr),
                    op: *op,
                    right: Box::new(right_expr),
                });
            }
        }

        Err(ParseError::SqlSyntax(format!("unsupported WHERE condition: {trimmed}")))
    }

    /// Parses a CREATE COLLECTION statement.
    ///
    /// Syntax:
    /// ```text
    /// CREATE COLLECTION [IF NOT EXISTS] name (
    ///     vector_name VECTOR_TYPE USING index_method [WITH (options...)],
    ///     ...
    /// );
    /// ```
    ///
    /// Vector types:
    /// - `VECTOR(dim)` - dense vectors
    /// - `SPARSE_VECTOR` or `SPARSE_VECTOR(max_dim)` - sparse vectors
    /// - `MULTI_VECTOR(dim)` - multi-vectors (e.g., ColBERT)
    /// - `BINARY_VECTOR(bits)` - binary vectors
    fn parse_create_collection(input: &str) -> ParseResult<Vec<Statement>> {
        let input = input.trim();
        let upper = input.to_uppercase();

        // Parse CREATE keyword
        if !upper.starts_with("CREATE") {
            return Err(ParseError::SqlSyntax("expected CREATE keyword".to_string()));
        }
        let after_create = input[6..].trim_start();
        let upper_after_create = after_create.to_uppercase();

        // Parse optional IF NOT EXISTS
        let (if_not_exists, after_if_not_exists) =
            if upper_after_create.starts_with("IF NOT EXISTS") {
                (true, after_create[13..].trim_start())
            } else {
                (false, after_create)
            };

        // Parse COLLECTION keyword
        let upper_rest = after_if_not_exists.to_uppercase();
        if !upper_rest.starts_with("COLLECTION") {
            return Err(ParseError::SqlSyntax(
                "expected COLLECTION keyword after CREATE".to_string(),
            ));
        }
        let after_collection = after_if_not_exists[10..].trim_start();

        // Parse collection name (identifier until '(' or whitespace)
        let name_end = after_collection
            .find(|c: char| c == '(' || c.is_whitespace())
            .unwrap_or(after_collection.len());
        let collection_name = &after_collection[..name_end];
        if collection_name.is_empty() {
            return Err(ParseError::SqlSyntax("expected collection name".to_string()));
        }
        let name = Identifier::new(collection_name.trim());

        // Find the opening and closing parentheses
        let after_name = after_collection[name_end..].trim_start();
        if !after_name.starts_with('(') {
            return Err(ParseError::SqlSyntax("expected '(' after collection name".to_string()));
        }

        // Find matching closing parenthesis
        let close_paren = Self::find_matching_paren(after_name, 0).ok_or_else(|| {
            ParseError::SqlSyntax("unclosed parenthesis in CREATE COLLECTION".to_string())
        })?;

        let defs_str = &after_name[1..close_paren];

        // Parse vector and payload field definitions
        let (vectors, payload_fields) = Self::parse_collection_definitions(defs_str)?;

        if vectors.is_empty() {
            return Err(ParseError::SqlSyntax(
                "CREATE COLLECTION requires at least one vector definition".to_string(),
            ));
        }

        let stmt = CreateCollectionStatement { if_not_exists, name, vectors, payload_fields };

        Ok(vec![Statement::CreateCollection(Box::new(stmt))])
    }

    /// Parses a Cypher-style CREATE statement for graph nodes and relationships.
    ///
    /// Syntax:
    /// ```text
    /// CREATE (n:Label {props})
    /// CREATE (n:Label {props}) RETURN n
    /// MATCH (a), (b) WHERE ... CREATE (a)-[:TYPE {props}]->(b)
    /// ```
    fn parse_cypher_create(input: &str) -> ParseResult<Vec<Statement>> {
        let input = input.trim().trim_end_matches(';');
        let upper = input.to_uppercase();

        // Check if we have a preceding MATCH clause
        let (match_clause, where_clause, create_start) = if upper.starts_with("MATCH") {
            // Find CREATE keyword position
            let create_pos = Self::find_keyword_pos(&upper, "CREATE")
                .ok_or_else(|| ParseError::InvalidPattern("expected CREATE keyword".to_string()))?;

            // Find WHERE position (if any, must be before CREATE)
            let where_pos = Self::find_keyword_pos(&upper[..create_pos], "WHERE");

            // Parse the MATCH pattern
            let match_end = where_pos.unwrap_or(create_pos);
            let pattern_str = &input[5..match_end].trim(); // 5 = "MATCH".len()
            let pattern = Self::parse_graph_pattern(pattern_str)?;

            // Parse WHERE clause if present
            let where_expr = if let Some(wp) = where_pos {
                let where_content = &input[wp + 5..create_pos]; // +5 for "WHERE"
                Some(Self::parse_where_expression(where_content.trim())?)
            } else {
                None
            };

            (Some(pattern), where_expr, create_pos)
        } else if upper.starts_with("CREATE") {
            (None, None, 0)
        } else {
            return Err(ParseError::InvalidPattern("expected CREATE or MATCH keyword".to_string()));
        };

        // Parse after CREATE
        let after_create = &input[create_start + 6..].trim_start(); // 6 = "CREATE".len()
        let upper_after_create = after_create.to_uppercase();

        // Find RETURN position if present
        let return_pos = Self::find_keyword_pos(&upper_after_create, "RETURN");

        // Parse CREATE patterns
        let patterns_end = return_pos.unwrap_or(after_create.len());
        let patterns_str = &after_create[..patterns_end].trim();
        let patterns = Self::parse_create_patterns(patterns_str)?;

        // Parse RETURN clause if present
        let return_clause = if let Some(rp) = return_pos {
            let return_content = &after_create[rp + 6..].trim(); // 6 = "RETURN".len()
            Self::parse_return_items(return_content)?
        } else {
            vec![]
        };

        let stmt = CreateGraphStatement { match_clause, where_clause, patterns, return_clause };

        Ok(vec![Statement::Create(Box::new(stmt))])
    }

    /// Parses CREATE patterns (nodes and relationships).
    fn parse_create_patterns(input: &str) -> ParseResult<Vec<CreatePattern>> {
        let input = input.trim();
        if input.is_empty() {
            return Err(ParseError::InvalidPattern("empty CREATE pattern".to_string()));
        }

        let mut patterns = Vec::new();
        let mut current = input;

        while !current.is_empty() {
            // Check if this looks like a path pattern (contains relationship)
            if Self::looks_like_path_pattern(current) {
                let (pattern, remaining) = Self::parse_create_path_pattern(current)?;
                patterns.push(pattern);
                current = remaining.trim();
            } else {
                // Parse as simple node pattern
                let (node, remaining) = Self::parse_create_node_pattern(current)?;
                patterns.push(node);
                current = remaining.trim();
            }

            // Skip comma separator
            if current.starts_with(',') {
                current = current[1..].trim();
            }
        }

        Ok(patterns)
    }

    /// Checks if the pattern looks like a path (contains relationship arrow).
    fn looks_like_path_pattern(input: &str) -> bool {
        // Look for relationship patterns: -[...]-> or <-[..]-
        input.contains("->") || input.contains("<-")
    }

    /// Parses a single CREATE node pattern like `(n:Label {props})`.
    fn parse_create_node_pattern(input: &str) -> ParseResult<(CreatePattern, &str)> {
        let (node, remaining) = Self::parse_node_pattern(input)?;

        let pattern = CreatePattern::Node {
            variable: node.variable,
            labels: node.labels,
            properties: node.properties.into_iter().map(|p| (p.name, p.value)).collect(),
        };

        Ok((pattern, remaining))
    }

    /// Parses a CREATE path pattern like `(a)-[:TYPE]->(b)`.
    fn parse_create_path_pattern(input: &str) -> ParseResult<(CreatePattern, &str)> {
        // Parse the start node
        let (start_node, after_start) = Self::parse_node_pattern(input)?;

        let start = if start_node.variable.is_some()
            && start_node.labels.is_empty()
            && start_node.properties.is_empty()
        {
            // Just a variable reference to an existing node
            CreateNodeRef::Variable(start_node.variable.clone().ok_or_else(|| {
                ParseError::InvalidPattern("expected variable in node reference".to_string())
            })?)
        } else {
            CreateNodeRef::New {
                variable: start_node.variable,
                labels: start_node.labels,
                properties: start_node.properties.into_iter().map(|p| (p.name, p.value)).collect(),
            }
        };

        // Parse path steps (relationships and destination nodes)
        let (steps, remaining) = Self::parse_create_path_steps(after_start.trim())?;

        if steps.is_empty() {
            return Err(ParseError::InvalidPattern(
                "expected at least one relationship in path pattern".to_string(),
            ));
        }

        let pattern = CreatePattern::Path { start, steps };

        Ok((pattern, remaining))
    }

    /// Parses path steps like `-[:TYPE]->(b)-[:TYPE2]->(c)`.
    fn parse_create_path_steps(input: &str) -> ParseResult<(Vec<CreatePathStep>, &str)> {
        let mut steps = Vec::new();
        let mut current = input;

        while !current.is_empty() {
            // Check if we're at a relationship start
            if !current.starts_with("-[") && !current.starts_with("<-[") {
                break; // No more relationships
            }

            // Parse edge pattern - this consumes the full edge including -> or <-
            let (edge, after_edge) = Self::parse_edge_pattern(current)?;
            let is_outgoing = edge.direction == EdgeDirection::Right;

            // The edge parsing already consumed the arrow, so after_edge points to the next node
            let after_edge = after_edge.trim();

            // Parse destination node
            let (dest_node, after_dest) = Self::parse_node_pattern(after_edge)?;

            let destination = if dest_node.variable.is_some()
                && dest_node.labels.is_empty()
                && dest_node.properties.is_empty()
            {
                CreateNodeRef::Variable(dest_node.variable.clone().ok_or_else(|| {
                    ParseError::InvalidPattern("expected variable in node reference".to_string())
                })?)
            } else {
                CreateNodeRef::New {
                    variable: dest_node.variable,
                    labels: dest_node.labels,
                    properties: dest_node
                        .properties
                        .into_iter()
                        .map(|p| (p.name, p.value))
                        .collect(),
                }
            };

            let rel_type = edge.edge_types.into_iter().next().ok_or_else(|| {
                ParseError::InvalidPattern("expected relationship type in CREATE".to_string())
            })?;

            steps.push(CreatePathStep {
                rel_variable: edge.variable,
                rel_type,
                rel_properties: edge.properties.into_iter().map(|p| (p.name, p.value)).collect(),
                outgoing: is_outgoing,
                destination,
            });

            current = after_dest.trim();
        }

        Ok((steps, current))
    }

    /// Parses a Cypher-style MERGE statement.
    ///
    /// Syntax:
    /// ```text
    /// MERGE (n:Label {key: value})
    /// MERGE (n:Label {key: value}) ON CREATE SET n.prop = val ON MATCH SET n.prop = val
    /// MATCH (a), (b) WHERE ... MERGE (a)-[:TYPE]->(b)
    /// ```
    fn parse_cypher_merge(input: &str) -> ParseResult<Vec<Statement>> {
        let input = input.trim().trim_end_matches(';');
        let upper = input.to_uppercase();

        // Check if we have a preceding MATCH clause
        let (match_clause, where_clause, merge_start) = if upper.starts_with("MATCH") {
            // Find MERGE keyword position
            let merge_pos = Self::find_keyword_pos(&upper, "MERGE")
                .ok_or_else(|| ParseError::InvalidPattern("expected MERGE keyword".to_string()))?;

            // Find WHERE position (if any, must be before MERGE)
            let where_pos = Self::find_keyword_pos(&upper[..merge_pos], "WHERE");

            // Parse the MATCH pattern
            let match_end = where_pos.unwrap_or(merge_pos);
            let pattern_str = &input[5..match_end].trim(); // 5 = "MATCH".len()
            let pattern = Self::parse_graph_pattern(pattern_str)?;

            // Parse WHERE clause if present
            let where_expr = if let Some(wp) = where_pos {
                let where_content = &input[wp + 5..merge_pos]; // +5 for "WHERE"
                Some(Self::parse_where_expression(where_content.trim())?)
            } else {
                None
            };

            (Some(pattern), where_expr, merge_pos)
        } else if upper.starts_with("MERGE") {
            (None, None, 0)
        } else {
            return Err(ParseError::InvalidPattern("expected MERGE or MATCH keyword".to_string()));
        };

        // Parse after MERGE
        let after_merge = &input[merge_start + 5..].trim_start(); // 5 = "MERGE".len()
        let upper_after_merge = after_merge.to_uppercase();

        // Find ON CREATE, ON MATCH, RETURN positions
        let on_create_pos = Self::find_keyword_pos(&upper_after_merge, "ON CREATE");
        let on_match_pos = Self::find_keyword_pos(&upper_after_merge, "ON MATCH");
        let return_pos = Self::find_keyword_pos(&upper_after_merge, "RETURN");

        // Pattern ends at first clause
        let pattern_end =
            on_create_pos.or(on_match_pos).or(return_pos).unwrap_or(after_merge.len());

        // Parse the MERGE pattern
        let pattern_str = &after_merge[..pattern_end].trim();
        let pattern = Self::parse_merge_pattern(pattern_str)?;

        // Parse ON CREATE SET clause
        let on_create = if let Some(pos) = on_create_pos {
            let start = pos + 9; // "ON CREATE".len()
            let end = on_match_pos.filter(|&p| p > pos).or(return_pos).unwrap_or(after_merge.len());
            Self::parse_set_actions(&after_merge[start..end])?
        } else {
            vec![]
        };

        // Parse ON MATCH SET clause
        let on_match = if let Some(pos) = on_match_pos {
            let start = pos + 8; // "ON MATCH".len()
            let end =
                on_create_pos.filter(|&p| p > pos).or(return_pos).unwrap_or(after_merge.len());
            Self::parse_set_actions(&after_merge[start..end])?
        } else {
            vec![]
        };

        // Parse RETURN clause if present
        let return_clause = if let Some(rp) = return_pos {
            let return_content = &after_merge[rp + 6..].trim(); // 6 = "RETURN".len()
            Self::parse_return_items(return_content)?
        } else {
            vec![]
        };

        let stmt = MergeGraphStatement {
            match_clause,
            where_clause,
            pattern,
            on_create,
            on_match,
            return_clause,
        };

        Ok(vec![Statement::Merge(Box::new(stmt))])
    }

    /// Parses a MERGE pattern.
    fn parse_merge_pattern(input: &str) -> ParseResult<MergePattern> {
        let input = input.trim();

        if Self::looks_like_path_pattern(input) {
            // Parse relationship pattern: (a)-[:TYPE]->(b)
            let (start_node, after_start) = Self::parse_node_pattern(input)?;

            // Start node should just be a variable reference
            let start = start_node.variable.ok_or_else(|| {
                ParseError::InvalidPattern(
                    "MERGE relationship requires named start node".to_string(),
                )
            })?;

            // Parse the relationship - parse_edge_pattern already consumes the arrow
            let after_start = after_start.trim();
            let (edge, after_edge) = Self::parse_edge_pattern(after_start)?;

            // Parse end node - after_edge already points past the arrow
            let (end_node, _) = Self::parse_node_pattern(after_edge.trim())?;

            let end = end_node.variable.ok_or_else(|| {
                ParseError::InvalidPattern("MERGE relationship requires named end node".to_string())
            })?;

            let rel_type = edge.edge_types.into_iter().next().ok_or_else(|| {
                ParseError::InvalidPattern("MERGE relationship requires type".to_string())
            })?;

            Ok(MergePattern::Relationship {
                start,
                rel_variable: edge.variable,
                rel_type,
                match_properties: edge.properties.into_iter().map(|p| (p.name, p.value)).collect(),
                end,
            })
        } else {
            // Parse node pattern: (n:Label {props})
            let (node, _) = Self::parse_node_pattern(input)?;

            let variable = node.variable.ok_or_else(|| {
                ParseError::InvalidPattern("MERGE node requires variable".to_string())
            })?;

            Ok(MergePattern::Node {
                variable,
                labels: node.labels,
                match_properties: node.properties.into_iter().map(|p| (p.name, p.value)).collect(),
            })
        }
    }

    /// Parses SET actions (for ON CREATE SET, ON MATCH SET).
    fn parse_set_actions(input: &str) -> ParseResult<Vec<SetAction>> {
        let input = input.trim();
        let upper = input.to_uppercase();

        // Must start with SET keyword
        if !upper.starts_with("SET") {
            return Ok(vec![]); // No SET clause
        }

        let after_set = &input[3..].trim(); // 3 = "SET".len()

        let mut actions = Vec::new();
        let mut current = *after_set;

        while !current.is_empty() {
            // Parse property assignment: var.prop = expr
            let (action, remaining) = Self::parse_single_set_action(current)?;
            actions.push(action);

            current = remaining.trim();
            if current.starts_with(',') {
                current = current[1..].trim();
            } else {
                break;
            }
        }

        Ok(actions)
    }

    /// Parses a single SET action.
    fn parse_single_set_action(input: &str) -> ParseResult<(SetAction, &str)> {
        let input = input.trim();

        // Parse variable name
        let dot_pos = input
            .find('.')
            .ok_or_else(|| ParseError::InvalidPattern("expected '.'".to_string()))?;

        let variable = Identifier::new(&input[..dot_pos]);
        let after_dot = &input[dot_pos + 1..];

        // Parse property name
        let eq_pos = after_dot
            .find('=')
            .ok_or_else(|| ParseError::InvalidPattern("expected '=' in SET clause".to_string()))?;

        let property = Identifier::new(after_dot[..eq_pos].trim());
        let after_eq = &after_dot[eq_pos + 1..];

        // Parse value expression - find end at comma or end of input
        let value_end = Self::find_expression_end(after_eq.trim());
        let value_str = &after_eq.trim()[..value_end];
        let value = Self::parse_where_expression(value_str.trim())?;

        let remaining = &after_eq.trim()[value_end..];

        Ok((SetAction::Property { variable, property, value }, remaining))
    }

    /// Finds the end of an expression (at comma or end of input).
    fn find_expression_end(input: &str) -> usize {
        let mut paren_depth: i32 = 0;
        let mut bracket_depth: i32 = 0;
        let mut brace_depth: i32 = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, c) in input.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' => escape_next = true,
                '\'' if !in_string => in_string = true,
                '\'' if in_string => in_string = false,
                '(' if !in_string => paren_depth += 1,
                ')' if !in_string => paren_depth = paren_depth.saturating_sub(1),
                '[' if !in_string => bracket_depth += 1,
                ']' if !in_string => bracket_depth = bracket_depth.saturating_sub(1),
                '{' if !in_string => brace_depth += 1,
                '}' if !in_string => brace_depth = brace_depth.saturating_sub(1),
                ',' if !in_string && paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                    return i;
                }
                _ => {}
            }
        }

        input.len()
    }

    /// Parses a Cypher-style SET statement.
    ///
    /// Syntax:
    /// ```text
    /// MATCH (n:Label) WHERE condition SET n.prop = value [, n.prop2 = value2] [RETURN n]
    /// MATCH (n:Label) SET n:NewLabel [RETURN n]
    /// ```
    fn parse_cypher_set(input: &str) -> ParseResult<Vec<Statement>> {
        let input = input.trim().trim_end_matches(';');
        let upper = input.to_uppercase();

        // Find the MATCH, WHERE, SET, and RETURN positions
        let set_pos = Self::find_keyword_pos(&upper, "SET")
            .ok_or_else(|| ParseError::InvalidPattern("expected SET keyword".to_string()))?;
        let where_pos = Self::find_keyword_pos(&upper[..set_pos], "WHERE");
        let return_pos = Self::find_keyword_pos(&upper, "RETURN");

        // Parse the MATCH pattern
        let match_end = where_pos.unwrap_or(set_pos);
        let pattern_str = &input[5..match_end].trim(); // 5 = "MATCH".len()
        let match_clause = Self::parse_graph_pattern(pattern_str)?;

        // Parse WHERE clause if present
        let where_clause = if let Some(wp) = where_pos {
            let where_content = &input[wp + 5..set_pos]; // +5 for "WHERE"
            Some(Self::parse_where_expression(where_content.trim())?)
        } else {
            None
        };

        // Parse SET clause
        let set_end = return_pos.unwrap_or(input.len());
        let set_content = &input[set_pos + 3..set_end].trim(); // +3 for "SET"
        let set_actions = Self::parse_set_items(set_content)?;

        // Parse RETURN clause if present
        let return_clause = if let Some(rp) = return_pos {
            let return_content = &input[rp + 6..].trim(); // 6 = "RETURN".len()
            Self::parse_return_items(return_content)?
        } else {
            vec![]
        };

        let mut stmt = SetGraphStatement::new(match_clause, set_actions);
        if let Some(w) = where_clause {
            stmt = stmt.with_where(w);
        }
        if !return_clause.is_empty() {
            stmt = stmt.with_return(return_clause);
        }

        Ok(vec![Statement::Set(Box::new(stmt))])
    }

    /// Parses SET items (property assignments and label additions).
    fn parse_set_items(input: &str) -> ParseResult<Vec<SetAction>> {
        let input = input.trim();
        if input.is_empty() {
            return Err(ParseError::InvalidPattern("SET clause cannot be empty".to_string()));
        }

        let mut actions = Vec::new();
        let items = Self::split_by_comma(input);

        for item in items {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }

            // Check if it's a label assignment (n:Label) or property assignment (n.prop = value)
            if item.contains('=') {
                // Property assignment: n.prop = value
                let eq_pos = item
                    .find('=')
                    .ok_or_else(|| ParseError::InvalidPattern("expected '='".to_string()))?;
                let left = item[..eq_pos].trim();
                let right = item[eq_pos + 1..].trim();

                // Parse left side: variable.property
                let dot_pos = left.find('.').ok_or_else(|| {
                    ParseError::InvalidPattern("expected 'variable.property' format".to_string())
                })?;
                let variable = Identifier::new(&left[..dot_pos]);
                let property = Identifier::new(&left[dot_pos + 1..]);

                // Parse value
                let value = Self::parse_where_expression(right)?;

                actions.push(SetAction::Property { variable, property, value });
            } else if item.contains(':') {
                // Label assignment: n:Label
                let colon_pos = item
                    .find(':')
                    .ok_or_else(|| ParseError::InvalidPattern("expected ':'".to_string()))?;
                let variable = Identifier::new(&item[..colon_pos]);
                let label = Identifier::new(&item[colon_pos + 1..]);

                actions.push(SetAction::Label { variable, label });
            } else {
                return Err(ParseError::InvalidPattern(format!(
                    "invalid SET item: {item}; expected property assignment or label"
                )));
            }
        }

        if actions.is_empty() {
            return Err(ParseError::InvalidPattern(
                "SET clause requires at least one action".to_string(),
            ));
        }

        Ok(actions)
    }

    /// Splits input by comma, respecting parentheses and quotes.
    fn split_by_comma(input: &str) -> Vec<&str> {
        let mut result = Vec::new();
        let mut start = 0;
        let mut paren_depth: i32 = 0;
        let mut bracket_depth: i32 = 0;
        let mut in_string = false;

        for (i, c) in input.char_indices() {
            match c {
                '\'' if !in_string => in_string = true,
                '\'' if in_string => in_string = false,
                '(' if !in_string => paren_depth += 1,
                ')' if !in_string => paren_depth = paren_depth.saturating_sub(1),
                '[' if !in_string => bracket_depth += 1,
                ']' if !in_string => bracket_depth = bracket_depth.saturating_sub(1),
                ',' if !in_string && paren_depth == 0 && bracket_depth == 0 => {
                    result.push(&input[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }

        result.push(&input[start..]);
        result
    }

    /// Parses a Cypher-style DELETE statement.
    ///
    /// Syntax:
    /// ```text
    /// MATCH (n:Label) WHERE condition DELETE n
    /// MATCH (n:Label) WHERE condition DETACH DELETE n
    /// MATCH (a)-[r]->(b) DELETE r
    /// ```
    fn parse_cypher_delete(input: &str) -> ParseResult<Vec<Statement>> {
        let input = input.trim().trim_end_matches(';');
        let upper = input.to_uppercase();

        // Check for DETACH DELETE
        let (detach, delete_pos) =
            if let Some(pos) = Self::find_keyword_pos(&upper, "DETACH DELETE") {
                (true, pos)
            } else if let Some(pos) = Self::find_keyword_pos(&upper, "DELETE") {
                (false, pos)
            } else {
                return Err(ParseError::InvalidPattern("expected DELETE keyword".to_string()));
            };

        let where_pos = Self::find_keyword_pos(&upper[..delete_pos], "WHERE");
        let return_pos = Self::find_keyword_pos(&upper, "RETURN");

        // Parse the MATCH pattern
        let match_end = where_pos.unwrap_or(delete_pos);
        let pattern_str = &input[5..match_end].trim(); // 5 = "MATCH".len()
        let match_clause = Self::parse_graph_pattern(pattern_str)?;

        // Parse WHERE clause if present
        let where_clause = if let Some(wp) = where_pos {
            let where_content = &input[wp + 5..delete_pos]; // +5 for "WHERE"
            Some(Self::parse_where_expression(where_content.trim())?)
        } else {
            None
        };

        // Parse DELETE variables
        let delete_keyword_len = if detach { 13 } else { 6 }; // "DETACH DELETE" or "DELETE"
        let delete_end = return_pos.unwrap_or(input.len());
        let delete_content = &input[delete_pos + delete_keyword_len..delete_end].trim();

        let variables: Vec<Identifier> = delete_content
            .split(',')
            .map(|s| Identifier::new(s.trim()))
            .filter(|id| !id.name.is_empty())
            .collect();

        if variables.is_empty() {
            return Err(ParseError::InvalidPattern(
                "DELETE requires at least one variable".to_string(),
            ));
        }

        // Parse RETURN clause if present
        let return_clause = if let Some(rp) = return_pos {
            let return_content = &input[rp + 6..].trim(); // 6 = "RETURN".len()
            Self::parse_return_items(return_content)?
        } else {
            vec![]
        };

        let mut stmt = if detach {
            DeleteGraphStatement::detach(match_clause, variables)
        } else {
            DeleteGraphStatement::new(match_clause, variables)
        };

        if let Some(w) = where_clause {
            stmt = stmt.with_where(w);
        }
        if !return_clause.is_empty() {
            stmt = stmt.with_return(return_clause);
        }

        Ok(vec![Statement::DeleteGraph(Box::new(stmt))])
    }

    /// Parses a Cypher-style REMOVE statement.
    ///
    /// Syntax:
    /// ```text
    /// MATCH (n:Label) WHERE condition REMOVE n.property
    /// MATCH (n:Label:Admin) REMOVE n:Admin
    /// ```
    fn parse_cypher_remove(input: &str) -> ParseResult<Vec<Statement>> {
        let input = input.trim().trim_end_matches(';');
        let upper = input.to_uppercase();

        let remove_pos = Self::find_keyword_pos(&upper, "REMOVE")
            .ok_or_else(|| ParseError::InvalidPattern("expected REMOVE keyword".to_string()))?;
        let where_pos = Self::find_keyword_pos(&upper[..remove_pos], "WHERE");
        let return_pos = Self::find_keyword_pos(&upper, "RETURN");

        // Parse the MATCH pattern
        let match_end = where_pos.unwrap_or(remove_pos);
        let pattern_str = &input[5..match_end].trim(); // 5 = "MATCH".len()
        let match_clause = Self::parse_graph_pattern(pattern_str)?;

        // Parse WHERE clause if present
        let where_clause = if let Some(wp) = where_pos {
            let where_content = &input[wp + 5..remove_pos]; // +5 for "WHERE"
            Some(Self::parse_where_expression(where_content.trim())?)
        } else {
            None
        };

        // Parse REMOVE items
        let remove_end = return_pos.unwrap_or(input.len());
        let remove_content = &input[remove_pos + 6..remove_end].trim(); // +6 for "REMOVE"

        let items = Self::parse_remove_items(remove_content)?;

        // Parse RETURN clause if present
        let return_clause = if let Some(rp) = return_pos {
            let return_content = &input[rp + 6..].trim(); // 6 = "RETURN".len()
            Self::parse_return_items(return_content)?
        } else {
            vec![]
        };

        let mut stmt = RemoveGraphStatement::new(match_clause, items);
        if let Some(w) = where_clause {
            stmt = stmt.with_where(w);
        }
        if !return_clause.is_empty() {
            stmt = stmt.with_return(return_clause);
        }

        Ok(vec![Statement::Remove(Box::new(stmt))])
    }

    /// Parses REMOVE items (property removal and label removal).
    fn parse_remove_items(input: &str) -> ParseResult<Vec<RemoveItem>> {
        let input = input.trim();
        if input.is_empty() {
            return Err(ParseError::InvalidPattern("REMOVE clause cannot be empty".to_string()));
        }

        let mut items = Vec::new();
        let parts = Self::split_by_comma(input);

        for part in parts {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            if part.contains('.') {
                // Property removal: n.property
                let dot_pos = part
                    .find('.')
                    .ok_or_else(|| ParseError::InvalidPattern("expected '.'".to_string()))?;
                let variable = Identifier::new(&part[..dot_pos]);
                let property = Identifier::new(&part[dot_pos + 1..]);
                items.push(RemoveItem::Property { variable, property });
            } else if part.contains(':') {
                // Label removal: n:Label
                let colon_pos = part
                    .find(':')
                    .ok_or_else(|| ParseError::InvalidPattern("expected ':'".to_string()))?;
                let variable = Identifier::new(&part[..colon_pos]);
                let label = Identifier::new(&part[colon_pos + 1..]);
                items.push(RemoveItem::Label { variable, label });
            } else {
                return Err(ParseError::InvalidPattern(format!(
                    "invalid REMOVE item: {part}; expected property or label"
                )));
            }
        }

        if items.is_empty() {
            return Err(ParseError::InvalidPattern(
                "REMOVE clause requires at least one item".to_string(),
            ));
        }

        Ok(items)
    }

    /// Parses collection definitions (vectors and payload fields).
    ///
    /// Supports both syntaxes:
    /// - New syntax: `title TEXT, VECTOR text_embedding DIMENSION 1536`
    /// - Legacy syntax: `dense VECTOR(768) USING hnsw`
    fn parse_collection_definitions(
        input: &str,
    ) -> ParseResult<(Vec<VectorDef>, Vec<PayloadFieldDef>)> {
        let input = input.trim();
        if input.is_empty() {
            return Ok((vec![], vec![]));
        }

        let mut vectors = Vec::new();
        let mut payload_fields = Vec::new();
        let mut current = String::new();
        let mut paren_depth: i32 = 0;

        // Split by comma, respecting parentheses
        for c in input.chars() {
            match c {
                '(' => {
                    paren_depth += 1;
                    current.push(c);
                }
                ')' => {
                    paren_depth = paren_depth.saturating_sub(1);
                    current.push(c);
                }
                ',' if paren_depth == 0 => {
                    if !current.trim().is_empty() {
                        Self::parse_collection_item(
                            current.trim(),
                            &mut vectors,
                            &mut payload_fields,
                        )?;
                    }
                    current.clear();
                }
                _ => current.push(c),
            }
        }

        // Don't forget the last definition
        if !current.trim().is_empty() {
            Self::parse_collection_item(current.trim(), &mut vectors, &mut payload_fields)?;
        }

        Ok((vectors, payload_fields))
    }

    /// Parses a single collection item (either a vector or a payload field).
    fn parse_collection_item(
        input: &str,
        vectors: &mut Vec<VectorDef>,
        payload_fields: &mut Vec<PayloadFieldDef>,
    ) -> ParseResult<()> {
        let input = input.trim();
        let upper = input.to_uppercase();

        // Check for new-style VECTOR declaration: VECTOR name DIMENSION dim
        if upper.starts_with("VECTOR ") {
            let vector = Self::parse_new_style_vector_def(input)?;
            vectors.push(vector);
            return Ok(());
        }

        // Check if it's a legacy vector definition (name VECTOR(dim), SPARSE_VECTOR, etc.)
        if Self::is_legacy_vector_def(input) {
            let vector = Self::parse_single_vector_def(input)?;
            vectors.push(vector);
            return Ok(());
        }

        // Otherwise, it's a payload field definition: name TYPE [INDEXED]
        let field = Self::parse_payload_field(input)?;
        payload_fields.push(field);
        Ok(())
    }

    /// Checks if the input is a legacy vector definition (name VECTOR(...)).
    fn is_legacy_vector_def(input: &str) -> bool {
        let upper = input.to_uppercase();
        // Find where the type starts (after the name)
        let name_end = input.find(|c: char| c.is_whitespace()).unwrap_or(input.len());
        let after_name = upper[name_end..].trim_start();
        after_name.starts_with("VECTOR")
            || after_name.starts_with("SPARSE_VECTOR")
            || after_name.starts_with("MULTI_VECTOR")
            || after_name.starts_with("BINARY_VECTOR")
    }

    /// Parses a new-style vector definition: `VECTOR name DIMENSION dim [USING method] [WITH (...)]`.
    fn parse_new_style_vector_def(input: &str) -> ParseResult<VectorDef> {
        let input = input.trim();
        let upper = input.to_uppercase();

        // Skip "VECTOR " prefix
        if !upper.starts_with("VECTOR ") {
            return Err(ParseError::SqlSyntax("expected VECTOR keyword".to_string()));
        }
        let after_vector = input[7..].trim_start();

        // Parse vector name
        let name_end = after_vector.find(|c: char| c.is_whitespace()).unwrap_or(after_vector.len());
        let name_str = &after_vector[..name_end];
        if name_str.is_empty() {
            return Err(ParseError::SqlSyntax("expected vector name after VECTOR".to_string()));
        }
        let name = Identifier::new(name_str);

        let after_name = after_vector[name_end..].trim_start();
        let upper_after_name = after_name.to_uppercase();

        // Parse DIMENSION
        if !upper_after_name.starts_with("DIMENSION") {
            return Err(ParseError::SqlSyntax(
                "expected DIMENSION keyword after vector name".to_string(),
            ));
        }
        let after_dimension = after_name[9..].trim_start();

        // Parse dimension value
        let dim_end =
            after_dimension.find(|c: char| c.is_whitespace()).unwrap_or(after_dimension.len());
        let dim_str = &after_dimension[..dim_end];
        let dimension = dim_str
            .trim()
            .parse::<u32>()
            .map_err(|_| ParseError::SqlSyntax(format!("invalid DIMENSION value: {dim_str}")))?;

        let vector_type = VectorTypeDef::Vector { dimension };

        // Parse optional USING and WITH clauses
        let rest = after_dimension[dim_end..].trim_start();
        let upper_rest = rest.to_uppercase();

        let (using_method, after_using) = if upper_rest.starts_with("USING") {
            let after_using_kw = rest[5..].trim_start();
            let method_end =
                after_using_kw.find(|c: char| c.is_whitespace()).unwrap_or(after_using_kw.len());
            let method = &after_using_kw[..method_end];
            (Some(method.to_lowercase()), after_using_kw[method_end..].trim_start())
        } else {
            (None, rest)
        };

        let upper_after_using = after_using.to_uppercase();
        let with_options = if upper_after_using.starts_with("WITH") {
            let after_with = after_using[4..].trim_start();
            Self::parse_with_options(after_with)?
        } else {
            vec![]
        };

        Ok(VectorDef { name, vector_type, using: using_method, with_options })
    }

    /// Parses a payload field definition: `name TYPE [INDEXED]`.
    fn parse_payload_field(input: &str) -> ParseResult<PayloadFieldDef> {
        let input = input.trim();

        // Parse field name
        let name_end = input.find(|c: char| c.is_whitespace()).unwrap_or(input.len());
        let name_str = &input[..name_end];
        if name_str.is_empty() {
            return Err(ParseError::SqlSyntax("expected field name".to_string()));
        }
        let name = Identifier::new(name_str);

        let after_name = input[name_end..].trim_start();
        let upper_after_name = after_name.to_uppercase();

        // Parse data type
        let (data_type, after_type) = Self::parse_payload_data_type(&upper_after_name, after_name)?;

        // Check for INDEXED keyword
        let upper_after_type = after_type.to_uppercase();
        let indexed = upper_after_type.trim().starts_with("INDEXED");

        Ok(PayloadFieldDef { name, data_type, indexed })
    }

    /// Parses a payload data type (TEXT, INTEGER, FLOAT, BOOLEAN, JSON, BYTEA).
    fn parse_payload_data_type<'a>(
        upper: &str,
        original: &'a str,
    ) -> ParseResult<(DataType, &'a str)> {
        if upper.starts_with("TEXT") {
            return Ok((DataType::Text, &original[4..]));
        }
        if upper.starts_with("INTEGER") {
            return Ok((DataType::Integer, &original[7..]));
        }
        if upper.starts_with("INT") {
            return Ok((DataType::Integer, &original[3..]));
        }
        if upper.starts_with("BIGINT") {
            return Ok((DataType::BigInt, &original[6..]));
        }
        if upper.starts_with("FLOAT") {
            return Ok((DataType::Real, &original[5..]));
        }
        if upper.starts_with("REAL") {
            return Ok((DataType::Real, &original[4..]));
        }
        if upper.starts_with("DOUBLE") {
            return Ok((DataType::DoublePrecision, &original[6..]));
        }
        if upper.starts_with("BOOLEAN") || upper.starts_with("BOOL") {
            let len = if upper.starts_with("BOOLEAN") { 7 } else { 4 };
            return Ok((DataType::Boolean, &original[len..]));
        }
        if upper.starts_with("JSON") {
            return Ok((DataType::Json, &original[4..]));
        }
        if upper.starts_with("JSONB") {
            return Ok((DataType::Jsonb, &original[5..]));
        }
        if upper.starts_with("BYTEA") {
            return Ok((DataType::Bytea, &original[5..]));
        }
        if upper.starts_with("TIMESTAMP") {
            return Ok((DataType::Timestamp, &original[9..]));
        }
        if upper.starts_with("DATE") {
            return Ok((DataType::Date, &original[4..]));
        }
        if upper.starts_with("UUID") {
            return Ok((DataType::Uuid, &original[4..]));
        }
        // VARCHAR(n)
        if upper.starts_with("VARCHAR") {
            let after_varchar = original[7..].trim_start();
            if after_varchar.starts_with('(') {
                let close = after_varchar.find(')').ok_or_else(|| {
                    ParseError::SqlSyntax("unclosed parenthesis in VARCHAR".to_string())
                })?;
                let len_str = &after_varchar[1..close];
                let len = len_str.trim().parse::<u32>().map_err(|_| {
                    ParseError::SqlSyntax(format!("invalid VARCHAR length: {len_str}"))
                })?;
                return Ok((DataType::Varchar(Some(len)), &after_varchar[close + 1..]));
            }
            return Ok((DataType::Varchar(None), &original[7..]));
        }

        Err(ParseError::SqlSyntax(format!(
            "expected data type (TEXT, INTEGER, FLOAT, BOOLEAN, JSON, BYTEA, etc.), found: {}",
            &original[..original.len().min(20)]
        )))
    }

    /// Parses the vector definitions inside CREATE COLLECTION parentheses.
    /// This is kept for backwards compatibility.
    /// TODO(v1.0): Remove after deprecation period for old syntax.
    #[allow(dead_code)]
    fn parse_vector_definitions(input: &str) -> ParseResult<Vec<VectorDef>> {
        let input = input.trim();
        if input.is_empty() {
            return Ok(vec![]);
        }

        let mut vectors = Vec::new();
        let mut current = String::new();
        let mut paren_depth: i32 = 0;

        // Split by comma, respecting parentheses
        for c in input.chars() {
            match c {
                '(' => {
                    paren_depth += 1;
                    current.push(c);
                }
                ')' => {
                    paren_depth = paren_depth.saturating_sub(1);
                    current.push(c);
                }
                ',' if paren_depth == 0 => {
                    if !current.trim().is_empty() {
                        vectors.push(Self::parse_single_vector_def(current.trim())?);
                    }
                    current.clear();
                }
                _ => current.push(c),
            }
        }

        // Don't forget the last definition
        if !current.trim().is_empty() {
            vectors.push(Self::parse_single_vector_def(current.trim())?);
        }

        Ok(vectors)
    }

    /// Parses a single vector definition.
    ///
    /// Format: `name TYPE [USING method] [WITH (options)]`
    fn parse_single_vector_def(input: &str) -> ParseResult<VectorDef> {
        let input = input.trim();

        // Find the name (first identifier)
        let name_end = input.find(|c: char| c.is_whitespace()).unwrap_or(input.len());
        let name_str = &input[..name_end];
        if name_str.is_empty() {
            return Err(ParseError::SqlSyntax("expected vector name in definition".to_string()));
        }
        let name = Identifier::new(name_str);

        let after_name = input[name_end..].trim_start();
        let upper_after_name = after_name.to_uppercase();

        // Parse the vector type
        let (vector_type, after_type) = Self::parse_vector_type(&upper_after_name, after_name)?;

        // Parse optional USING clause
        let after_type_trimmed = after_type.trim_start();
        let upper_after_type = after_type_trimmed.to_uppercase();
        let (using_method, after_using) = if upper_after_type.starts_with("USING") {
            let after_using_kw = after_type_trimmed[5..].trim_start();
            let upper_after_using_kw = after_using_kw.to_uppercase();

            // Find the method name (until WITH or end)
            let method_end =
                if let Some(with_pos) = Self::find_keyword_pos(&upper_after_using_kw, "WITH") {
                    with_pos
                } else {
                    after_using_kw.len()
                };

            let method = after_using_kw[..method_end].trim();
            (Some(method.to_lowercase()), after_using_kw[method_end..].trim_start())
        } else {
            (None, after_type_trimmed)
        };

        // Parse optional WITH clause
        let upper_after_using = after_using.to_uppercase();
        let with_options = if upper_after_using.starts_with("WITH") {
            let after_with = after_using[4..].trim_start();
            Self::parse_with_options(after_with)?
        } else {
            vec![]
        };

        Ok(VectorDef { name, vector_type, using: using_method, with_options })
    }

    /// Parses a vector type (VECTOR, SPARSE_VECTOR, MULTI_VECTOR, BINARY_VECTOR).
    fn parse_vector_type<'a>(
        upper: &str,
        original: &'a str,
    ) -> ParseResult<(VectorTypeDef, &'a str)> {
        // VECTOR(dim)
        if upper.starts_with("VECTOR") {
            let after_vector = original[6..].trim_start();
            if after_vector.starts_with('(') {
                let close = after_vector.find(')').ok_or_else(|| {
                    ParseError::SqlSyntax("unclosed parenthesis in VECTOR type".to_string())
                })?;
                let dim_str = &after_vector[1..close];
                let dimension = dim_str.trim().parse::<u32>().map_err(|_| {
                    ParseError::SqlSyntax(format!("invalid dimension in VECTOR: {dim_str}"))
                })?;
                return Ok((VectorTypeDef::Vector { dimension }, &after_vector[close + 1..]));
            }
            return Err(ParseError::SqlSyntax(
                "VECTOR type requires dimension: VECTOR(dim)".to_string(),
            ));
        }

        // SPARSE_VECTOR or SPARSE_VECTOR(max_dim)
        if upper.starts_with("SPARSE_VECTOR") {
            let after_sparse = original[13..].trim_start();
            if after_sparse.starts_with('(') {
                let close = after_sparse.find(')').ok_or_else(|| {
                    ParseError::SqlSyntax("unclosed parenthesis in SPARSE_VECTOR type".to_string())
                })?;
                let dim_str = &after_sparse[1..close];
                let max_dimension = Some(dim_str.trim().parse::<u32>().map_err(|_| {
                    ParseError::SqlSyntax(format!(
                        "invalid max dimension in SPARSE_VECTOR: {dim_str}"
                    ))
                })?);
                return Ok((
                    VectorTypeDef::SparseVector { max_dimension },
                    &after_sparse[close + 1..],
                ));
            }
            return Ok((VectorTypeDef::SparseVector { max_dimension: None }, after_sparse));
        }

        // MULTI_VECTOR(dim)
        if upper.starts_with("MULTI_VECTOR") {
            let after_multi = original[12..].trim_start();
            if after_multi.starts_with('(') {
                let close = after_multi.find(')').ok_or_else(|| {
                    ParseError::SqlSyntax("unclosed parenthesis in MULTI_VECTOR type".to_string())
                })?;
                let dim_str = &after_multi[1..close];
                let token_dim = dim_str.trim().parse::<u32>().map_err(|_| {
                    ParseError::SqlSyntax(format!("invalid dimension in MULTI_VECTOR: {dim_str}"))
                })?;
                return Ok((VectorTypeDef::MultiVector { token_dim }, &after_multi[close + 1..]));
            }
            return Err(ParseError::SqlSyntax(
                "MULTI_VECTOR type requires dimension: MULTI_VECTOR(dim)".to_string(),
            ));
        }

        // BINARY_VECTOR(bits)
        if upper.starts_with("BINARY_VECTOR") {
            let after_binary = original[13..].trim_start();
            if after_binary.starts_with('(') {
                let close = after_binary.find(')').ok_or_else(|| {
                    ParseError::SqlSyntax("unclosed parenthesis in BINARY_VECTOR type".to_string())
                })?;
                let bits_str = &after_binary[1..close];
                let bits = bits_str.trim().parse::<u32>().map_err(|_| {
                    ParseError::SqlSyntax(format!("invalid bits in BINARY_VECTOR: {bits_str}"))
                })?;
                return Ok((VectorTypeDef::BinaryVector { bits }, &after_binary[close + 1..]));
            }
            return Err(ParseError::SqlSyntax(
                "BINARY_VECTOR type requires bit count: BINARY_VECTOR(bits)".to_string(),
            ));
        }

        Err(ParseError::SqlSyntax(format!(
            "expected vector type (VECTOR, SPARSE_VECTOR, MULTI_VECTOR, BINARY_VECTOR), found: {}",
            &original[..original.len().min(20)]
        )))
    }

    /// Parses WITH options: `(key = 'value', key2 = 'value2', ...)`
    fn parse_with_options(input: &str) -> ParseResult<Vec<(String, String)>> {
        let input = input.trim();
        if !input.starts_with('(') {
            return Err(ParseError::SqlSyntax("expected '(' after WITH keyword".to_string()));
        }

        let close = input.find(')').ok_or_else(|| {
            ParseError::SqlSyntax("unclosed parenthesis in WITH options".to_string())
        })?;

        let options_str = &input[1..close];
        let mut options = Vec::new();

        // Split by comma and parse each key = value pair
        for pair in options_str.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }

            let eq_pos = pair.find('=').ok_or_else(|| {
                ParseError::SqlSyntax(format!("expected '=' in WITH option: {pair}"))
            })?;

            let key = pair[..eq_pos].trim().to_lowercase();
            let value_part = pair[eq_pos + 1..].trim();

            // Strip quotes from value if present
            let value = if (value_part.starts_with('\'') && value_part.ends_with('\''))
                || (value_part.starts_with('"') && value_part.ends_with('"'))
            {
                value_part[1..value_part.len() - 1].to_string()
            } else {
                value_part.to_string()
            };

            options.push((key, value));
        }

        Ok(options)
    }

    /// Parses a standalone MATCH statement (Cypher-style).
    ///
    /// Syntax:
    /// ```text
    /// MATCH <pattern>
    /// [WHERE <condition>]
    /// RETURN [DISTINCT] <return_items>
    /// [ORDER BY <expressions>]
    /// [SKIP <number>]
    /// [LIMIT <number>]
    /// ```
    fn parse_standalone_match(input: &str) -> ParseResult<Vec<Statement>> {
        let input = input.trim();
        let upper = input.to_uppercase();

        // Find MATCH keyword (should be at start)
        if !upper.starts_with("MATCH") {
            return Err(ParseError::InvalidPattern("expected MATCH keyword".to_string()));
        }

        // Skip "MATCH" keyword
        let after_match = input[5..].trim_start();

        // Find WHERE, RETURN, ORDER BY, SKIP, LIMIT positions
        let upper_after_match = after_match.to_uppercase();

        // Find RETURN (required)
        let return_pos = Self::find_keyword_pos(&upper_after_match, "RETURN").ok_or_else(|| {
            ParseError::InvalidPattern("MATCH requires RETURN clause".to_string())
        })?;

        // Find WHERE (optional, must be before RETURN)
        let where_pos = Self::find_keyword_pos(&upper_after_match[..return_pos], "WHERE");

        // Parse the graph pattern (everything between MATCH and WHERE/RETURN)
        let pattern_end = where_pos.unwrap_or(return_pos);
        let pattern_str = after_match[..pattern_end].trim();
        let pattern = Self::parse_graph_pattern(pattern_str)?;

        // Parse WHERE clause if present
        let where_clause = if let Some(wp) = where_pos {
            let where_content = &after_match[wp + 5..return_pos]; // +5 for "WHERE"
            Some(Self::parse_where_expression(where_content.trim())?)
        } else {
            None
        };

        // Parse after RETURN
        let after_return = after_match[return_pos + 6..].trim_start(); // +6 for "RETURN"
        let upper_after_return = after_return.to_uppercase();

        // Check for DISTINCT
        let (distinct, return_content_start) = if upper_after_return.starts_with("DISTINCT") {
            (true, 8) // "DISTINCT" is 8 chars
        } else {
            (false, 0)
        };

        let return_and_rest = after_return[return_content_start..].trim_start();
        let upper_return_rest = return_and_rest.to_uppercase();

        // Find ORDER BY, SKIP, LIMIT positions relative to RETURN items
        let order_by_pos = Self::find_keyword_pos(&upper_return_rest, "ORDER BY");
        let skip_pos = Self::find_keyword_pos(&upper_return_rest, "SKIP");
        let limit_pos = Self::find_keyword_pos(&upper_return_rest, "LIMIT");

        // The return items end at the first of ORDER BY, SKIP, LIMIT, or end of string
        let return_items_end = [order_by_pos, skip_pos, limit_pos]
            .iter()
            .filter_map(|&p| p)
            .min()
            .unwrap_or(return_and_rest.len());

        // Parse return items
        let return_items_str = return_and_rest[..return_items_end].trim();
        let return_items = Self::parse_return_items(return_items_str)?;

        // Parse ORDER BY
        let order_by = if let Some(obp) = order_by_pos {
            let order_end = [skip_pos, limit_pos]
                .iter()
                .filter_map(|&p| p)
                .min()
                .unwrap_or(return_and_rest.len());

            if order_end > obp + 8 {
                let order_content = &return_and_rest[obp + 8..order_end]; // +8 for "ORDER BY"
                Self::parse_order_by(order_content.trim())?
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // Parse SKIP
        let skip = if let Some(sp) = skip_pos {
            let skip_end = limit_pos.unwrap_or(return_and_rest.len());
            if skip_end > sp + 4 {
                let skip_content = &return_and_rest[sp + 4..skip_end]; // +4 for "SKIP"
                Some(Self::parse_limit_expr(skip_content.trim())?)
            } else {
                None
            }
        } else {
            None
        };

        // Parse LIMIT
        let limit = if let Some(lp) = limit_pos {
            let limit_content = &return_and_rest[lp + 5..]; // +5 for "LIMIT"
            let limit_end = limit_content.find(';').unwrap_or(limit_content.len());
            let limit_str = limit_content[..limit_end].trim();
            if limit_str.is_empty() {
                None
            } else {
                Some(Self::parse_limit_expr(limit_str)?)
            }
        } else {
            None
        };

        // Build the MatchStatement
        let match_stmt = MatchStatement {
            pattern,
            where_clause,
            return_clause: return_items,
            distinct,
            order_by,
            skip,
            limit,
        };

        Ok(vec![Statement::Match(Box::new(match_stmt))])
    }

    /// Finds the position of a keyword in a string (case-insensitive, word boundary).
    fn find_keyword_pos(input: &str, keyword: &str) -> Option<usize> {
        let mut search_from = 0;

        while let Some(pos) = input[search_from..].find(keyword) {
            let absolute_pos = search_from + pos;

            // Check word boundaries
            let before_ok =
                absolute_pos == 0 || !input.as_bytes()[absolute_pos - 1].is_ascii_alphanumeric();
            let after_ok = absolute_pos + keyword.len() >= input.len()
                || !input.as_bytes()[absolute_pos + keyword.len()].is_ascii_alphanumeric();

            if before_ok && after_ok {
                return Some(absolute_pos);
            }

            search_from = absolute_pos + keyword.len();
        }

        None
    }

    /// Parses a WHERE expression for standalone MATCH.
    fn parse_where_expression(input: &str) -> ParseResult<Expr> {
        // For simple expressions, try to parse them directly
        // This is a simplified parser - complex expressions go through SQL parser
        Self::parse_simple_expression(input)
    }

    /// Parses a simple expression (for WHERE and RETURN clauses).
    fn parse_simple_expression(input: &str) -> ParseResult<Expr> {
        let input = input.trim();

        if input.is_empty() {
            return Err(ParseError::InvalidPattern("empty expression".to_string()));
        }

        // Check for list comprehension or list literal: [...]
        if input.starts_with('[') {
            return Self::parse_list_or_comprehension(input);
        }

        // Check for AND/OR at the top level
        if let Some(and_pos) = Self::find_top_level_keyword(input, " AND ") {
            let left = Self::parse_simple_expression(&input[..and_pos])?;
            let right = Self::parse_simple_expression(&input[and_pos + 5..])?;
            return Ok(Expr::BinaryOp {
                left: Box::new(left),
                op: crate::ast::BinaryOp::And,
                right: Box::new(right),
            });
        }

        if let Some(or_pos) = Self::find_top_level_keyword(input, " OR ") {
            let left = Self::parse_simple_expression(&input[..or_pos])?;
            let right = Self::parse_simple_expression(&input[or_pos + 4..])?;
            return Ok(Expr::BinaryOp {
                left: Box::new(left),
                op: crate::ast::BinaryOp::Or,
                right: Box::new(right),
            });
        }

        // Check for comparison operators
        let comparisons = [
            ("<>", crate::ast::BinaryOp::NotEq),
            ("!=", crate::ast::BinaryOp::NotEq),
            ("<=", crate::ast::BinaryOp::LtEq),
            (">=", crate::ast::BinaryOp::GtEq),
            ("<", crate::ast::BinaryOp::Lt),
            (">", crate::ast::BinaryOp::Gt),
            ("=", crate::ast::BinaryOp::Eq),
        ];

        for (op_str, op) in &comparisons {
            if let Some(pos) = Self::find_top_level_operator(input, op_str) {
                let left = Self::parse_simple_expression(&input[..pos])?;
                let right = Self::parse_simple_expression(&input[pos + op_str.len()..])?;
                return Ok(Expr::BinaryOp {
                    left: Box::new(left),
                    op: *op,
                    right: Box::new(right),
                });
            }
        }

        // Parse as a simple value or column reference
        Ok(Self::parse_property_value(input))
    }

    /// Parses a list literal or list comprehension.
    ///
    /// List literal: `[expr1, expr2, ...]`
    /// List comprehension: `[x IN list WHERE predicate | expression]`
    fn parse_list_or_comprehension(input: &str) -> ParseResult<Expr> {
        let input = input.trim();

        // Must start with [ and end with ]
        if !input.starts_with('[') || !input.ends_with(']') {
            return Err(ParseError::InvalidPattern(
                "list expression must be enclosed in brackets".to_string(),
            ));
        }

        // Get the content inside brackets
        let inner = &input[1..input.len() - 1].trim();

        if inner.is_empty() {
            // Empty list literal
            return Ok(Expr::ListLiteral(vec![]));
        }

        // Check if this is a list comprehension: look for " IN " at top level
        if let Some(in_pos) = Self::find_top_level_keyword(inner, " IN ") {
            // This is a list comprehension: [variable IN list_expr WHERE pred | transform]
            let variable_str = inner[..in_pos].trim();

            // Variable must be a simple identifier
            if variable_str.is_empty()
                || !variable_str.chars().all(|c| c.is_alphanumeric() || c == '_')
            {
                // Not a valid variable name, might be a list literal with IN operator
                // Fall through to list literal parsing
            } else {
                // Parse the rest after " IN "
                let after_in = inner[in_pos + 4..].trim();

                // Find WHERE and | positions at top level
                let where_pos = Self::find_top_level_keyword(after_in, " WHERE ");
                let pipe_pos = Self::find_top_level_operator(after_in, "|");

                // Determine the boundaries of list_expr, filter, and transform
                let (list_expr_str, filter_str, transform_str) = match (where_pos, pipe_pos) {
                    (Some(w), Some(p)) if w < p => {
                        // [x IN list WHERE pred | transform]
                        let list_part = after_in[..w].trim();
                        let filter_part = after_in[w + 7..p].trim();
                        let transform_part = after_in[p + 1..].trim();
                        (list_part, Some(filter_part), Some(transform_part))
                    }
                    (Some(_), Some(_)) => {
                        // Invalid: pipe before or at WHERE position - treat as list literal
                        return Self::parse_list_literal(inner);
                    }
                    (Some(w), None) => {
                        // [x IN list WHERE pred] - filter only
                        let list_part = after_in[..w].trim();
                        let filter_part = after_in[w + 7..].trim();
                        (list_part, Some(filter_part), None)
                    }
                    (None, Some(p)) => {
                        // [x IN list | transform] - transform only
                        let list_part = after_in[..p].trim();
                        let transform_part = after_in[p + 1..].trim();
                        (list_part, None, Some(transform_part))
                    }
                    (None, None) => {
                        // [x IN list] - just iteration, returns elements unchanged
                        (after_in, None, None)
                    }
                };

                // Parse the components
                let list_expr = Self::parse_simple_expression(list_expr_str)?;

                let filter_predicate = if let Some(f) = filter_str {
                    if f.is_empty() {
                        None
                    } else {
                        Some(Box::new(Self::parse_simple_expression(f)?))
                    }
                } else {
                    None
                };

                let transform_expr = if let Some(t) = transform_str {
                    if t.is_empty() {
                        None
                    } else {
                        Some(Box::new(Self::parse_simple_expression(t)?))
                    }
                } else {
                    None
                };

                return Ok(Expr::ListComprehension {
                    variable: Identifier::new(variable_str),
                    list_expr: Box::new(list_expr),
                    filter_predicate,
                    transform_expr,
                });
            }
        }

        // Not a comprehension, parse as list literal
        Self::parse_list_literal(inner)
    }

    /// Parses a list literal content (without brackets).
    fn parse_list_literal(content: &str) -> ParseResult<Expr> {
        let content = content.trim();

        if content.is_empty() {
            return Ok(Expr::ListLiteral(vec![]));
        }

        // Split by comma at top level
        let mut elements = Vec::new();
        let mut current = String::new();
        let mut depth = 0;
        let mut in_string = false;
        let mut string_char = '"';

        for c in content.chars() {
            if in_string {
                if c == string_char {
                    in_string = false;
                }
                current.push(c);
                continue;
            }

            match c {
                '\'' | '"' => {
                    in_string = true;
                    string_char = c;
                    current.push(c);
                }
                '(' | '[' => {
                    depth += 1;
                    current.push(c);
                }
                ')' | ']' => {
                    depth -= 1;
                    current.push(c);
                }
                ',' if depth == 0 => {
                    let trimmed = current.trim();
                    if !trimmed.is_empty() {
                        elements.push(Self::parse_simple_expression(trimmed)?);
                    }
                    current.clear();
                }
                _ => current.push(c),
            }
        }

        // Don't forget the last element
        let trimmed = current.trim();
        if !trimmed.is_empty() {
            elements.push(Self::parse_simple_expression(trimmed)?);
        }

        Ok(Expr::ListLiteral(elements))
    }

    /// Finds a keyword at the top level (not inside parentheses).
    fn find_top_level_keyword(input: &str, keyword: &str) -> Option<usize> {
        let upper = input.to_uppercase();
        let keyword_upper = keyword.to_uppercase();
        let mut depth: i32 = 0;
        let mut i = 0;
        let bytes = input.as_bytes();

        while i < input.len() {
            if bytes[i] == b'(' {
                depth += 1;
            } else if bytes[i] == b')' {
                depth = depth.saturating_sub(1);
            } else if depth == 0 && upper[i..].starts_with(&keyword_upper) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// Finds an operator at the top level (not inside parentheses or strings).
    fn find_top_level_operator(input: &str, op: &str) -> Option<usize> {
        let mut depth: i32 = 0;
        let mut in_string = false;
        let mut string_char = '"';
        let bytes = input.as_bytes();
        let op_bytes = op.as_bytes();

        if op.len() > input.len() {
            return None;
        }

        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i];

            if in_string {
                if c == string_char as u8 {
                    in_string = false;
                }
                i += 1;
                continue;
            }

            match c {
                b'\'' | b'"' => {
                    in_string = true;
                    string_char = c as char;
                }
                b'(' => depth += 1,
                b')' => depth = depth.saturating_sub(1),
                _ if depth == 0 && i + op.len() <= bytes.len() => {
                    if &bytes[i..i + op.len()] == op_bytes {
                        return Some(i);
                    }
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// Parses return items from a RETURN clause.
    fn parse_return_items(input: &str) -> ParseResult<Vec<ReturnItem>> {
        let input = input.trim();

        if input.is_empty() {
            return Err(ParseError::InvalidPattern("empty RETURN clause".to_string()));
        }

        // Check for wildcard
        if input == "*" {
            return Ok(vec![ReturnItem::Wildcard]);
        }

        // Split by comma (respecting parentheses)
        let mut items = Vec::new();
        let mut current = String::new();
        let mut depth: i32 = 0;

        for c in input.chars() {
            match c {
                '(' => {
                    depth += 1;
                    current.push(c);
                }
                ')' => {
                    depth = depth.saturating_sub(1);
                    current.push(c);
                }
                ',' if depth == 0 => {
                    if !current.trim().is_empty() {
                        items.push(Self::parse_return_item(current.trim())?);
                    }
                    current.clear();
                }
                _ => current.push(c),
            }
        }

        if !current.trim().is_empty() {
            items.push(Self::parse_return_item(current.trim())?);
        }

        if items.is_empty() {
            return Err(ParseError::InvalidPattern("empty RETURN clause".to_string()));
        }

        Ok(items)
    }

    /// Parses a single return item.
    fn parse_return_item(input: &str) -> ParseResult<ReturnItem> {
        let input = input.trim();

        if input == "*" {
            return Ok(ReturnItem::Wildcard);
        }

        // Check for AS alias
        let upper = input.to_uppercase();
        if let Some(as_pos) = Self::find_top_level_keyword(&upper, " AS ") {
            let expr_str = &input[..as_pos];
            let alias_str = &input[as_pos + 4..]; // +4 for " AS "
            let expr = Self::parse_simple_expression(expr_str.trim())?;
            let alias = Identifier::new(alias_str.trim());
            return Ok(ReturnItem::Expr { expr, alias: Some(alias) });
        }

        // Just an expression
        let expr = Self::parse_simple_expression(input)?;
        Ok(ReturnItem::Expr { expr, alias: None })
    }

    /// Parses an ORDER BY clause.
    fn parse_order_by(input: &str) -> ParseResult<Vec<OrderByExpr>> {
        let input = input.trim();

        if input.is_empty() {
            return Ok(vec![]);
        }

        let mut orders = Vec::new();
        let mut current = String::new();
        let mut depth: i32 = 0;

        for c in input.chars() {
            match c {
                '(' => {
                    depth += 1;
                    current.push(c);
                }
                ')' => {
                    depth = depth.saturating_sub(1);
                    current.push(c);
                }
                ',' if depth == 0 => {
                    if !current.trim().is_empty() {
                        orders.push(Self::parse_order_item(current.trim())?);
                    }
                    current.clear();
                }
                _ => current.push(c),
            }
        }

        if !current.trim().is_empty() {
            orders.push(Self::parse_order_item(current.trim())?);
        }

        Ok(orders)
    }

    /// Parses a single ORDER BY item.
    fn parse_order_item(input: &str) -> ParseResult<OrderByExpr> {
        let input = input.trim();
        let upper = input.to_uppercase();

        // Check for ASC/DESC at the end
        let (expr_str, asc) = if upper.ends_with(" DESC") {
            (&input[..input.len() - 5], false)
        } else if upper.ends_with(" ASC") {
            (&input[..input.len() - 4], true)
        } else {
            (input, true) // Default to ASC
        };

        let expr = Self::parse_simple_expression(expr_str.trim())?;

        Ok(OrderByExpr { expr: Box::new(expr), asc, nulls_first: None })
    }

    /// Parses a LIMIT/SKIP expression (must be an integer).
    fn parse_limit_expr(input: &str) -> ParseResult<Expr> {
        let input = input.trim();

        // Try to parse as integer
        if let Ok(n) = input.parse::<i64>() {
            return Ok(Expr::integer(n));
        }

        // Try as parameter
        if let Some(rest) = input.strip_prefix('$') {
            if let Ok(n) = rest.parse::<u32>() {
                return Ok(Expr::Parameter(ParameterRef::Positional(n)));
            }
            return Ok(Expr::Parameter(ParameterRef::Named(rest.to_string())));
        }

        Err(ParseError::InvalidPattern(format!("invalid LIMIT/SKIP value: {input}")))
    }

    /// Parses a single extended SQL statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or contains multiple statements.
    pub fn parse_single(input: &str) -> ParseResult<Statement> {
        let mut stmts = Self::parse(input)?;
        if stmts.len() != 1 {
            return Err(ParseError::SqlSyntax(format!(
                "expected 1 statement, found {}",
                stmts.len()
            )));
        }
        Ok(stmts.remove(0))
    }

    /// Extracts MATCH and OPTIONAL MATCH clauses from the SQL and returns the modified SQL.
    ///
    /// Returns a tuple of (modified SQL, required MATCH patterns, OPTIONAL MATCH patterns per statement).
    ///
    /// The expected syntax is:
    /// ```sql
    /// SELECT ... FROM MATCH (pattern) OPTIONAL MATCH (opt_pattern1) OPTIONAL MATCH (opt_pattern2) WHERE ...
    /// ```
    ///
    /// Where OPTIONAL MATCH clauses follow the required MATCH and apply to it.
    fn extract_match_clauses(
        input: &str,
    ) -> ParseResult<(String, Vec<GraphPattern>, Vec<Vec<GraphPattern>>)> {
        let mut result = String::with_capacity(input.len());
        let mut match_patterns: Vec<GraphPattern> = Vec::new();
        let mut optional_patterns: Vec<Vec<GraphPattern>> = Vec::new();
        let mut remaining = input;

        loop {
            // Find the next OPTIONAL MATCH or MATCH keyword
            let optional_pos = Self::find_optional_match_keyword(remaining);
            let match_pos = Self::find_match_keyword(remaining);

            // Determine which comes first
            match (optional_pos, match_pos) {
                (Some(opt_pos), Some(m_pos)) if opt_pos < m_pos => {
                    // OPTIONAL MATCH comes before a regular MATCH
                    // This shouldn't normally happen in valid syntax, but handle it
                    result.push_str(&remaining[..opt_pos]);

                    // Skip "OPTIONAL" and whitespace before "MATCH"
                    let after_optional = &remaining[opt_pos + 8..]; // "OPTIONAL" = 8 chars
                    let after_optional_trimmed = after_optional.trim_start();
                    let whitespace_len = after_optional.len() - after_optional_trimmed.len();

                    // Skip "MATCH"
                    let after_match = &after_optional[whitespace_len + 5..];
                    let end_pos = Self::find_match_end(after_match);

                    let pattern_str = after_match[..end_pos].trim();
                    let pattern = Self::parse_graph_pattern(pattern_str)?;

                    // Attach to the last required MATCH, or create a standalone entry
                    if let Some(last_optionals) = optional_patterns.last_mut() {
                        last_optionals.push(pattern);
                    } else {
                        // No required MATCH yet - this is unusual but handle it
                        optional_patterns.push(vec![pattern]);
                    }

                    remaining = &after_match[end_pos..];
                }
                (Some(opt_pos), None) => {
                    // Only OPTIONAL MATCH found (no more required MATCH)
                    result.push_str(&remaining[..opt_pos]);

                    let after_optional = &remaining[opt_pos + 8..];
                    let after_optional_trimmed = after_optional.trim_start();
                    let whitespace_len = after_optional.len() - after_optional_trimmed.len();

                    let after_match = &after_optional[whitespace_len + 5..];
                    let end_pos = Self::find_match_end(after_match);

                    let pattern_str = after_match[..end_pos].trim();
                    let pattern = Self::parse_graph_pattern(pattern_str)?;

                    // Attach to the last required MATCH
                    if let Some(last_optionals) = optional_patterns.last_mut() {
                        last_optionals.push(pattern);
                    } else {
                        // No required MATCH yet - unusual but handle it
                        optional_patterns.push(vec![pattern]);
                    }

                    remaining = &after_match[end_pos..];
                }
                (_, Some(m_pos)) => {
                    // Regular MATCH comes first (or there's no OPTIONAL MATCH in this part)
                    result.push_str(&remaining[..m_pos]);

                    let after_match = &remaining[m_pos + 5..]; // Skip "MATCH"
                    let end_pos = Self::find_match_end(after_match);

                    let pattern_str = after_match[..end_pos].trim();
                    let pattern = Self::parse_graph_pattern(pattern_str)?;

                    // Store the required match pattern
                    match_patterns.push(pattern);
                    // Create a new (empty) vector for optional patterns that will follow this MATCH
                    optional_patterns.push(Vec::new());

                    remaining = &after_match[end_pos..];
                }
                (None, None) => {
                    // No more MATCH clauses
                    break;
                }
            }
        }

        result.push_str(remaining);

        Ok((result, match_patterns, optional_patterns))
    }

    /// Finds the position of the OPTIONAL MATCH keyword pair (case-insensitive, word boundary).
    fn find_optional_match_keyword(input: &str) -> Option<usize> {
        let input_upper = input.to_uppercase();
        let mut search_from = 0;

        while let Some(pos) = input_upper[search_from..].find("OPTIONAL") {
            let absolute_pos = search_from + pos;

            // Check word boundaries for OPTIONAL
            let before_ok =
                absolute_pos == 0 || !input.as_bytes()[absolute_pos - 1].is_ascii_alphanumeric();
            let after_ok = absolute_pos + 8 >= input.len()
                || !input.as_bytes()[absolute_pos + 8].is_ascii_alphanumeric();

            if before_ok && after_ok {
                // Check if MATCH follows after whitespace
                let after_optional = &input_upper[absolute_pos + 8..];
                let after_optional_trimmed = after_optional.trim_start();

                if after_optional_trimmed.starts_with("MATCH") {
                    // Check word boundary after MATCH
                    let match_start =
                        absolute_pos + 8 + (after_optional.len() - after_optional_trimmed.len());
                    let match_end = match_start + 5;
                    let match_after_ok = match_end >= input.len()
                        || !input.as_bytes()[match_end].is_ascii_alphanumeric();

                    if match_after_ok {
                        return Some(absolute_pos);
                    }
                }
            }

            search_from = absolute_pos + 8;
        }

        None
    }

    /// Finds the position of the MATCH keyword (case-insensitive, word boundary).
    ///
    /// This function skips "MATCH" when it's part of "OPTIONAL MATCH".
    fn find_match_keyword(input: &str) -> Option<usize> {
        let input_upper = input.to_uppercase();
        let mut search_from = 0;

        while let Some(pos) = input_upper[search_from..].find("MATCH") {
            let absolute_pos = search_from + pos;

            // Check word boundaries
            let before_ok =
                absolute_pos == 0 || !input.as_bytes()[absolute_pos - 1].is_ascii_alphanumeric();
            let after_ok = absolute_pos + 5 >= input.len()
                || !input.as_bytes()[absolute_pos + 5].is_ascii_alphanumeric();

            if before_ok && after_ok {
                // Check if this is part of "OPTIONAL MATCH" - look backwards for OPTIONAL
                let is_optional_match = Self::is_preceded_by_optional(&input_upper, absolute_pos);

                if !is_optional_match {
                    return Some(absolute_pos);
                }
            }

            search_from = absolute_pos + 5;
        }

        None
    }

    /// Checks if a MATCH at the given position is preceded by OPTIONAL (with whitespace).
    fn is_preceded_by_optional(input_upper: &str, match_pos: usize) -> bool {
        if match_pos < 8 {
            // "OPTIONAL" is 8 chars, so not enough room
            return false;
        }

        // Look backwards from match_pos, skipping whitespace
        let before_match = &input_upper[..match_pos];
        let trimmed = before_match.trim_end();

        // Check if it ends with "OPTIONAL"
        if trimmed.len() >= 8 && trimmed.ends_with("OPTIONAL") {
            // Check word boundary before OPTIONAL
            let optional_start = trimmed.len() - 8;
            if optional_start == 0 {
                return true;
            }
            let byte_before = trimmed.as_bytes()[optional_start - 1];
            return !byte_before.is_ascii_alphanumeric();
        }

        false
    }

    /// Finds the end of a MATCH clause.
    ///
    /// A MATCH clause ends at the next SQL keyword (WHERE, ORDER BY, etc.),
    /// another MATCH or OPTIONAL MATCH clause, or a semicolon.
    fn find_match_end(input: &str) -> usize {
        let input_upper = input.to_uppercase();

        let keywords = [
            "WHERE",
            "ORDER",
            "GROUP",
            "HAVING",
            "LIMIT",
            "OFFSET",
            "UNION",
            "INTERSECT",
            "EXCEPT",
            "OPTIONAL", // Stop at OPTIONAL MATCH
            "MATCH",    // Stop at the next MATCH (for multiple OPTIONAL MATCHes)
        ];

        let mut min_pos = input.len();

        for keyword in &keywords {
            if let Some(pos) = input_upper.find(keyword) {
                // Check word boundary
                let before_ok = pos == 0 || !input.as_bytes()[pos - 1].is_ascii_alphanumeric();
                let after_ok = pos + keyword.len() >= input.len()
                    || !input.as_bytes()[pos + keyword.len()].is_ascii_alphanumeric();
                if before_ok && after_ok && pos < min_pos {
                    min_pos = pos;
                }
            }
        }

        // Also check for semicolon
        if let Some(pos) = input.find(';') {
            if pos < min_pos {
                min_pos = pos;
            }
        }

        min_pos
    }

    /// Pre-processes vector operators to temporary function calls.
    ///
    /// Converts `a <-> b` to `__VEC_EUCLIDEAN__(a, b)` which sqlparser can parse.
    fn preprocess_vector_ops(input: &str) -> String {
        // Only allocate if we find operators to replace
        if !input.contains("<->")
            && !input.contains("<=>")
            && !input.contains("<#>")
            && !input.contains("<##>")
        {
            return input.to_string();
        }

        let mut result = input.to_string();
        result = Self::replace_vector_op(&result, "<->", "__VEC_EUCLIDEAN__");
        result = Self::replace_vector_op(&result, "<=>", "__VEC_COSINE__");
        // Handle <##> (MaxSim) before <#> (InnerProduct) since <##> contains <#>
        result = Self::replace_vector_op(&result, "<##>", "__VEC_MAXSIM__");
        result = Self::replace_vector_op(&result, "<#>", "__VEC_INNER__");
        result
    }

    /// Replaces a vector operator with a function call.
    ///
    /// Converts `expr1 <op> expr2` to `__FUNC__(expr1, expr2)`.
    fn replace_vector_op(input: &str, op: &str, func_name: &str) -> String {
        let chars: Vec<char> = input.chars().collect();
        let op_chars: Vec<char> = op.chars().collect();

        // Find the operator position
        let Some(op_pos) = Self::find_operator(&chars, &op_chars) else {
            return input.to_string();
        };

        // Find the left operand (go backwards to find start)
        let left_end = op_pos;
        let left_start = Self::find_expr_start(&chars, left_end);

        // Find the right operand (go forwards to find end)
        let right_start = op_pos + op_chars.len();
        let right_end = Self::find_expr_end(&chars, right_start);

        // Build the result
        let mut result = String::with_capacity(input.len() + 64);

        // Add everything before the left operand
        result.extend(&chars[..left_start]);

        // Add the function call
        result.push_str(func_name);
        result.push('(');
        let left_expr: String = chars[left_start..left_end].iter().collect();
        result.push_str(left_expr.trim());
        result.push_str(", ");
        let right_expr: String = chars[right_start..right_end].iter().collect();
        result.push_str(right_expr.trim());
        result.push(')');

        // Add everything after the right operand
        result.extend(&chars[right_end..]);

        // Recursively process for any remaining operators
        Self::replace_vector_op(&result, op, func_name)
    }

    /// Finds the position of an operator in the character array.
    fn find_operator(chars: &[char], op_chars: &[char]) -> Option<usize> {
        (0..chars.len()).find(|&i| {
            i + op_chars.len() <= chars.len() && chars[i..i + op_chars.len()] == op_chars[..]
        })
    }

    /// Finds the start of an expression going backwards from `end`.
    fn find_expr_start(chars: &[char], end: usize) -> usize {
        let mut pos = end;
        let mut paren_depth = 0;

        // Skip trailing whitespace before the operator
        while pos > 0 && chars[pos - 1].is_whitespace() {
            pos -= 1;
        }

        // Go backwards to find the start of the expression
        while pos > 0 {
            let c = chars[pos - 1];
            match c {
                ')' => {
                    paren_depth += 1;
                    pos -= 1;
                }
                '(' => {
                    if paren_depth > 0 {
                        paren_depth -= 1;
                        pos -= 1;
                    } else {
                        break;
                    }
                }
                ']' => {
                    // Handle array subscripts and literals (including nested arrays)
                    let mut bracket_depth = 1;
                    pos -= 1;
                    while pos > 0 && bracket_depth > 0 {
                        match chars[pos - 1] {
                            ']' => bracket_depth += 1,
                            '[' => bracket_depth -= 1,
                            _ => {}
                        }
                        pos -= 1;
                    }
                }
                _ if c.is_alphanumeric() || c == '_' || c == '.' || c == '$' => {
                    pos -= 1;
                }
                _ if paren_depth > 0 => {
                    pos -= 1;
                }
                ',' | ';' | '=' | '>' | '<' | '+' | '-' | '*' | '/' => {
                    break;
                }
                _ if c.is_whitespace() => {
                    // At a space, we need to check if this separates tokens
                    // Stop here - the expression starts after this whitespace
                    break;
                }
                _ => {
                    break;
                }
            }
        }

        pos
    }

    /// Finds the end of an expression going forwards from `start`.
    fn find_expr_end(chars: &[char], start: usize) -> usize {
        let mut pos = start;
        let mut paren_depth = 0;

        // Skip leading whitespace after the operator
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

        // Go forwards to find the end of the expression
        while pos < chars.len() {
            let c = chars[pos];
            match c {
                '(' => {
                    paren_depth += 1;
                    pos += 1;
                }
                ')' => {
                    if paren_depth > 0 {
                        paren_depth -= 1;
                        pos += 1;
                    } else {
                        break;
                    }
                }
                '[' => {
                    // Handle array literals and subscripts (including nested arrays)
                    let mut bracket_depth = 1;
                    pos += 1;
                    while pos < chars.len() && bracket_depth > 0 {
                        match chars[pos] {
                            '[' => bracket_depth += 1,
                            ']' => bracket_depth -= 1,
                            _ => {}
                        }
                        pos += 1;
                    }
                }
                _ if c.is_alphanumeric() || c == '_' || c == '.' || c == '$' => {
                    pos += 1;
                }
                _ if paren_depth > 0 => {
                    pos += 1;
                }
                _ if c.is_whitespace() => {
                    // At a space, the expression ends here
                    break;
                }
                ',' | ';' | '<' | '>' | '=' | '+' | '-' | '*' | '/' => {
                    break;
                }
                _ => {
                    break;
                }
            }
        }

        pos
    }

    /// Restores vector operators from function call markers.
    fn restore_vector_ops(stmt: &mut Statement) {
        match stmt {
            Statement::Select(select) => Self::restore_vector_ops_in_select(select),
            Statement::Update(update) => {
                if let Some(ref mut expr) = update.where_clause {
                    Self::restore_vector_ops_in_expr(expr);
                }
            }
            Statement::Delete(delete) => {
                if let Some(ref mut expr) = delete.where_clause {
                    Self::restore_vector_ops_in_expr(expr);
                }
            }
            Statement::Explain(inner) => Self::restore_vector_ops(inner),
            _ => {}
        }
    }

    /// Restores vector operators in a SELECT statement.
    fn restore_vector_ops_in_select(select: &mut SelectStatement) {
        // Process projection
        for item in &mut select.projection {
            if let crate::ast::SelectItem::Expr { expr, .. } = item {
                Self::restore_vector_ops_in_expr(expr);
            }
        }

        // Process WHERE clause
        if let Some(ref mut expr) = select.where_clause {
            Self::restore_vector_ops_in_expr(expr);
        }

        // Process ORDER BY
        for order in &mut select.order_by {
            Self::restore_vector_ops_in_expr(&mut order.expr);
        }

        // Process HAVING
        if let Some(ref mut expr) = select.having {
            Self::restore_vector_ops_in_expr(expr);
        }
    }

    /// Restores vector operators in an expression.
    ///
    /// Converts function calls like `__VEC_EUCLIDEAN__(a, b)` back to `a <-> b`.
    fn restore_vector_ops_in_expr(expr: &mut Expr) {
        // First, recursively process child expressions
        match expr {
            Expr::BinaryOp { left, right, .. } => {
                Self::restore_vector_ops_in_expr(left);
                Self::restore_vector_ops_in_expr(right);
            }
            Expr::UnaryOp { operand, .. } => {
                Self::restore_vector_ops_in_expr(operand);
            }
            Expr::Function(func) => {
                // Process function arguments first
                for arg in &mut func.args {
                    Self::restore_vector_ops_in_expr(arg);
                }
            }
            Expr::Case(case) => {
                if let Some(ref mut operand) = case.operand {
                    Self::restore_vector_ops_in_expr(operand);
                }
                for (cond, result) in &mut case.when_clauses {
                    Self::restore_vector_ops_in_expr(cond);
                    Self::restore_vector_ops_in_expr(result);
                }
                if let Some(ref mut else_result) = case.else_result {
                    Self::restore_vector_ops_in_expr(else_result);
                }
            }
            Expr::HybridSearch { components, .. } => {
                for comp in components {
                    Self::restore_vector_ops_in_expr(&mut comp.distance_expr);
                }
            }
            _ => {}
        }

        // Now check if this expression is a vector function call and convert it
        Self::convert_vector_function(expr);
    }

    /// Converts vector function calls to binary operators.
    ///
    /// Transforms:
    /// - `__VEC_EUCLIDEAN__(a, b)` to `Expr::BinaryOp { left: a, op: EuclideanDistance, right: b }`
    /// - `COSINE_SIMILARITY(vector_name, query)` to `Expr::BinaryOp { left: vector_name, op: CosineDistance, right: query }`
    /// - `EUCLIDEAN_DISTANCE(vector_name, query)` to `Expr::BinaryOp { left: vector_name, op: EuclideanDistance, right: query }`
    /// - `INNER_PRODUCT(vector_name, query)` to `Expr::BinaryOp { left: vector_name, op: InnerProduct, right: query }`
    /// - `MAXSIM(vector_name, query)` to `Expr::BinaryOp { left: vector_name, op: MaxSim, right: query }`
    fn convert_vector_function(expr: &mut Expr) {
        let replacement = if let Expr::Function(func) = expr {
            let func_name = func.name.name().map(|id| id.name.as_str()).unwrap_or("");

            // Check for HYBRID function first
            if func_name.eq_ignore_ascii_case("HYBRID") || func_name.eq_ignore_ascii_case("RRF") {
                Self::parse_hybrid_function(func, func_name.eq_ignore_ascii_case("RRF"))
            } else {
                // Check for vector distance operators (both internal and user-facing function names)
                let op = Self::parse_distance_function_name(func_name);

                if let Some(op) = op {
                    if func.args.len() == 2 {
                        let mut args = std::mem::take(&mut func.args);
                        let right = args.pop().expect("checked len");
                        let left = args.pop().expect("checked len");
                        Some(Expr::BinaryOp { left: Box::new(left), op, right: Box::new(right) })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        } else {
            None
        };

        if let Some(new_expr) = replacement {
            *expr = new_expr;
        }
    }

    /// Parses a distance function name and returns the corresponding binary operator.
    ///
    /// Supports both internal marker names (from preprocessing) and user-facing function names.
    fn parse_distance_function_name(func_name: &str) -> Option<BinaryOp> {
        // Internal marker names from preprocessing
        if func_name == "__VEC_EUCLIDEAN__" {
            return Some(BinaryOp::EuclideanDistance);
        }
        if func_name == "__VEC_COSINE__" {
            return Some(BinaryOp::CosineDistance);
        }
        if func_name == "__VEC_INNER__" {
            return Some(BinaryOp::InnerProduct);
        }
        if func_name == "__VEC_MAXSIM__" {
            return Some(BinaryOp::MaxSim);
        }

        // User-facing similarity function names (case-insensitive)
        let upper = func_name.to_uppercase();
        match upper.as_str() {
            // Cosine similarity/distance functions
            "COSINE_SIMILARITY" | "COSINE_DISTANCE" | "COS_DISTANCE" | "COS_SIM" => {
                Some(BinaryOp::CosineDistance)
            }
            // Euclidean distance functions
            "EUCLIDEAN_DISTANCE" | "L2_DISTANCE" | "EUCLIDEAN" | "L2" => {
                Some(BinaryOp::EuclideanDistance)
            }
            // Inner product functions
            "INNER_PRODUCT" | "DOT_PRODUCT" | "DOT" => Some(BinaryOp::InnerProduct),
            // MaxSim for multi-vectors
            "MAXSIM" | "MAX_SIM" => Some(BinaryOp::MaxSim),
            _ => None,
        }
    }

    /// Parses a HYBRID or RRF function call into a HybridSearch expression.
    ///
    /// Syntax: `HYBRID(expr1, weight1, expr2, weight2, ...)`
    /// Or: `RRF(expr1, expr2, ...)` (uses RRF with k=60)
    fn parse_hybrid_function(func: &mut crate::ast::FunctionCall, is_rrf: bool) -> Option<Expr> {
        use crate::ast::{HybridCombinationMethod, HybridSearchComponent};

        if is_rrf {
            // RRF(expr1, expr2, ...) - each expr has equal weight
            if func.args.is_empty() {
                return None;
            }

            let components: Vec<HybridSearchComponent> = std::mem::take(&mut func.args)
                .into_iter()
                .map(|arg| HybridSearchComponent::new(arg, 1.0))
                .collect();

            Some(Expr::HybridSearch { components, method: HybridCombinationMethod::RRF { k: 60 } })
        } else {
            // HYBRID(expr1, weight1, expr2, weight2, ...)
            // Must have even number of args, at least 4
            if func.args.len() < 4 || func.args.len() % 2 != 0 {
                return None;
            }

            let mut components = Vec::new();
            let args = std::mem::take(&mut func.args);

            let mut iter = args.into_iter();
            while let Some(distance_expr) = iter.next() {
                let weight_expr = iter.next()?;

                // Extract weight value
                let weight = Self::extract_weight(&weight_expr)?;

                components.push(HybridSearchComponent::new(distance_expr, weight));
            }

            Some(Expr::HybridSearch { components, method: HybridCombinationMethod::WeightedSum })
        }
    }

    /// Extracts a numeric weight from an expression.
    fn extract_weight(expr: &Expr) -> Option<f64> {
        match expr {
            Expr::Literal(crate::ast::Literal::Float(f)) => Some(*f),
            Expr::Literal(crate::ast::Literal::Integer(i)) => Some(*i as f64),
            _ => None,
        }
    }

    /// Adds a MATCH clause to a statement.
    fn add_match_clause(stmt: &mut Statement, pattern: GraphPattern) {
        match stmt {
            Statement::Select(select) => {
                select.match_clause = Some(pattern);
            }
            Statement::Update(update) => {
                update.match_clause = Some(pattern);
            }
            Statement::Delete(delete) => {
                delete.match_clause = Some(pattern);
            }
            _ => {}
        }
    }

    /// Adds an OPTIONAL MATCH clause to a statement.
    ///
    /// OPTIONAL MATCH clauses are joined using LEFT OUTER JOIN semantics.
    fn add_optional_match_clause(stmt: &mut Statement, pattern: GraphPattern) {
        if let Statement::Select(select) = stmt {
            select.optional_match_clauses.push(pattern);
        }
        // OPTIONAL MATCH is not supported for UPDATE/DELETE - just ignore
    }

    /// Parses a graph pattern string.
    fn parse_graph_pattern(input: &str) -> ParseResult<GraphPattern> {
        let input = input.trim();
        if input.is_empty() {
            return Err(ParseError::InvalidPattern("empty pattern".to_string()));
        }

        // Most patterns have 1-2 paths
        let mut paths = Vec::with_capacity(2);
        let mut current = input;

        while !current.is_empty() {
            let (path, remaining) = Self::parse_path_pattern(current)?;
            paths.push(path);

            current = remaining.trim();
            if current.starts_with(',') {
                current = current[1..].trim();
            }
        }

        if paths.is_empty() {
            return Err(ParseError::InvalidPattern("no paths in pattern".to_string()));
        }

        Ok(GraphPattern::new(paths))
    }

    /// Parses a path pattern.
    fn parse_path_pattern(input: &str) -> ParseResult<(PathPattern, &str)> {
        let (start, remaining) = Self::parse_node_pattern(input)?;
        let mut path = PathPattern::node(start);
        let mut current = remaining;

        loop {
            current = current.trim_start();

            // Check for edge pattern
            if current.starts_with('-') || current.starts_with('<') {
                let (edge, after_edge) = Self::parse_edge_pattern(current)?;
                let (node, after_node) = Self::parse_node_pattern(after_edge.trim_start())?;
                path = path.then(edge, node);
                current = after_node;
            } else {
                break;
            }
        }

        Ok((path, current))
    }

    /// Parses a node pattern: `(variable:Label:Label2 {prop: value})`.
    fn parse_node_pattern(input: &str) -> ParseResult<(NodePattern, &str)> {
        let input = input.trim_start();

        if !input.starts_with('(') {
            return Err(ParseError::InvalidPattern(format!(
                "expected '(' at start of node pattern, found: {}",
                input.chars().next().unwrap_or('?')
            )));
        }

        let close_paren = Self::find_matching_paren(input, 0)
            .ok_or_else(|| ParseError::InvalidPattern("unclosed node pattern".to_string()))?;

        let inner = &input[1..close_paren];
        let remaining = &input[close_paren + 1..];

        let node = Self::parse_node_inner(inner)?;
        Ok((node, remaining))
    }

    /// Parses the inner content of a node pattern.
    fn parse_node_inner(input: &str) -> ParseResult<NodePattern> {
        let input = input.trim();

        if input.is_empty() {
            return Ok(NodePattern::anonymous());
        }

        let mut variable = None;
        // Most nodes have 1-2 labels
        let mut labels = Vec::with_capacity(2);
        // Most nodes have 0-3 properties
        let mut properties = Vec::with_capacity(2);

        let mut current = input;

        // Parse variable (before first colon or brace)
        if !current.starts_with(':') && !current.starts_with('{') {
            let end = current.find([':', '{', ' ']).unwrap_or(current.len());
            let var_name = &current[..end];
            if !var_name.is_empty() {
                variable = Some(Identifier::new(var_name));
            }
            current = &current[end..];
        }

        // Parse labels (each starts with :)
        while current.starts_with(':') {
            current = &current[1..]; // Skip ':'
            let end = current.find([':', '{', ' ', ')']).unwrap_or(current.len());
            let label = &current[..end];
            if !label.is_empty() {
                labels.push(Identifier::new(label));
            }
            current = current[end..].trim_start();
        }

        // Parse properties (in braces)
        if current.starts_with('{') {
            let close_brace = current
                .find('}')
                .ok_or_else(|| ParseError::InvalidPattern("unclosed properties".to_string()))?;
            let props_str = &current[1..close_brace];
            properties = Self::parse_properties(props_str)?;
        }

        Ok(NodePattern { variable, labels, properties })
    }

    /// Parses an edge pattern: `-[variable:TYPE*min..max]->` or `<-[...]-`.
    fn parse_edge_pattern(input: &str) -> ParseResult<(EdgePattern, &str)> {
        let input = input.trim_start();

        // Determine direction based on starting characters
        let (direction, bracket_start) = if input.starts_with("<-[") {
            (EdgeDirection::Left, 2)
        } else if input.starts_with("-[") {
            // Could be -> or undirected, check ending
            (EdgeDirection::Right, 1) // Will verify later
        } else {
            return Err(ParseError::InvalidPattern(format!(
                "expected edge pattern, found: {}",
                &input[..input.len().min(10)]
            )));
        };

        // Find the closing bracket
        let bracket_end = input[bracket_start + 1..]
            .find(']')
            .map(|p| p + bracket_start + 1)
            .ok_or_else(|| ParseError::InvalidPattern("unclosed edge pattern".to_string()))?;

        let inner = &input[bracket_start + 1..bracket_end];
        let after_bracket = &input[bracket_end + 1..];

        // Determine actual direction from ending
        let (actual_direction, remaining) = if let Some(rest) = after_bracket.strip_prefix("->") {
            (EdgeDirection::Right, rest)
        } else if let Some(rest) = after_bracket.strip_prefix('-') {
            if direction == EdgeDirection::Left {
                (EdgeDirection::Left, rest)
            } else {
                (EdgeDirection::Undirected, rest)
            }
        } else {
            return Err(ParseError::InvalidPattern("invalid edge ending".to_string()));
        };

        let edge = Self::parse_edge_inner(inner, actual_direction)?;
        Ok((edge, remaining))
    }

    /// Parses the inner content of an edge pattern.
    fn parse_edge_inner(input: &str, direction: EdgeDirection) -> ParseResult<EdgePattern> {
        let input = input.trim();

        let mut variable = None;
        // Most edges have 1-2 types
        let mut edge_types = Vec::with_capacity(2);
        let mut length = EdgeLength::Single;
        // Most edges have 0-2 properties
        let mut properties = Vec::with_capacity(2);

        if input.is_empty() {
            return Ok(EdgePattern { direction, variable, edge_types, properties, length });
        }

        let mut current = input;

        // Parse variable (before first colon, asterisk, or brace)
        if !current.starts_with(':') && !current.starts_with('*') && !current.starts_with('{') {
            let end = current.find([':', '*', '{', ' ']).unwrap_or(current.len());
            let var_name = &current[..end];
            if !var_name.is_empty() {
                variable = Some(Identifier::new(var_name));
            }
            current = &current[end..];
        }

        // Parse edge types (each starts with : or |)
        while current.starts_with(':') || current.starts_with('|') {
            current = &current[1..]; // Skip ':' or '|'
            let end = current.find(['|', '*', '{', ' ', ']']).unwrap_or(current.len());
            let edge_type = &current[..end];
            if !edge_type.is_empty() {
                edge_types.push(Identifier::new(edge_type));
            }
            current = current[end..].trim_start();
        }

        // Parse length (*min..max, *n, or *)
        if current.starts_with('*') {
            current = &current[1..];
            length = Self::parse_edge_length(current)?;

            // Skip past the length specification
            let end = current.find(['{', ' ', ']']).unwrap_or(current.len());
            current = current[end..].trim_start();
        }

        // Parse properties (in braces)
        if current.starts_with('{') {
            let close_brace = current.find('}').ok_or_else(|| {
                ParseError::InvalidPattern("unclosed edge properties".to_string())
            })?;
            let props_str = &current[1..close_brace];
            properties = Self::parse_properties(props_str)?;
        }

        Ok(EdgePattern { direction, variable, edge_types, properties, length })
    }

    /// Parses edge length specification.
    fn parse_edge_length(input: &str) -> ParseResult<EdgeLength> {
        let input = input.trim();

        if input.is_empty()
            || input.starts_with('{')
            || input.starts_with(' ')
            || input.starts_with(']')
        {
            return Ok(EdgeLength::Any);
        }

        // Check for range (min..max)
        if let Some(range_pos) = input.find("..") {
            let before = &input[..range_pos];
            let after_start = range_pos + 2;
            let after_end = input[after_start..]
                .find(|c: char| !c.is_ascii_digit())
                .map_or(input.len(), |p| after_start + p);
            let after = &input[after_start..after_end];

            let min = if before.is_empty() {
                None
            } else {
                Some(before.parse::<u32>().map_err(|_| {
                    ParseError::InvalidPattern(format!("invalid min in range: {before}"))
                })?)
            };

            let max = if after.is_empty() {
                None
            } else {
                Some(after.parse::<u32>().map_err(|_| {
                    ParseError::InvalidPattern(format!("invalid max in range: {after}"))
                })?)
            };

            return Ok(EdgeLength::Range { min, max });
        }

        // Check for exact number
        let num_end = input.find(|c: char| !c.is_ascii_digit()).unwrap_or(input.len());
        let num_str = &input[..num_end];

        if !num_str.is_empty() {
            let n = num_str.parse::<u32>().map_err(|_| {
                ParseError::InvalidPattern(format!("invalid edge length: {num_str}"))
            })?;
            return Ok(EdgeLength::Exact(n));
        }

        Ok(EdgeLength::Any)
    }

    /// Parses properties from a property string like `name: 'Alice', age: 30`.
    fn parse_properties(input: &str) -> ParseResult<Vec<PropertyCondition>> {
        let input = input.trim();
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Estimate properties count from comma count + 1
        let estimated_count = input.matches(',').count() + 1;
        let mut properties = Vec::with_capacity(estimated_count);

        for pair in input.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }

            let colon_pos = pair
                .find(':')
                .ok_or_else(|| ParseError::InvalidPattern(format!("invalid property: {pair}")))?;

            let name = pair[..colon_pos].trim();
            let value_str = pair[colon_pos + 1..].trim();

            let value = Self::parse_property_value(value_str);

            properties.push(PropertyCondition { name: Identifier::new(name), value });
        }

        Ok(properties)
    }

    /// Parses a property value.
    fn parse_property_value(input: &str) -> Expr {
        let input = input.trim();

        // String literal
        if (input.starts_with('\'') && input.ends_with('\''))
            || (input.starts_with('"') && input.ends_with('"'))
        {
            let s = &input[1..input.len() - 1];
            return Expr::string(s);
        }

        // Boolean
        if input.eq_ignore_ascii_case("true") {
            return Expr::boolean(true);
        }
        if input.eq_ignore_ascii_case("false") {
            return Expr::boolean(false);
        }

        // Null
        if input.eq_ignore_ascii_case("null") {
            return Expr::null();
        }

        // Parameter
        if let Some(name) = input.strip_prefix('$') {
            if let Ok(n) = name.parse::<u32>() {
                return Expr::Parameter(ParameterRef::Positional(n));
            }
            return Expr::Parameter(ParameterRef::Named(name.to_string()));
        }

        // Integer
        if let Ok(i) = input.parse::<i64>() {
            return Expr::integer(i);
        }

        // Float
        if let Ok(f) = input.parse::<f64>() {
            return Expr::float(f);
        }

        // Identifier (column reference)
        Expr::column(QualifiedName::simple(input))
    }

    /// Finds the matching closing parenthesis.
    fn find_matching_paren(input: &str, open_pos: usize) -> Option<usize> {
        let bytes = input.as_bytes();
        let mut depth = 0;
        let mut in_string = false;
        let mut string_char = b'"';

        for (i, &byte) in bytes.iter().enumerate().skip(open_pos) {
            if in_string {
                if byte == string_char && (i == 0 || bytes[i - 1] != b'\\') {
                    in_string = false;
                }
                continue;
            }

            match byte {
                b'\'' | b'"' => {
                    in_string = true;
                    string_char = byte;
                }
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }

        None
    }
}

/// Parses a shortest path pattern from a string.
///
/// Handles the following syntaxes:
/// - `SHORTEST PATH (a)-[*]->(b)` - unweighted shortest path
/// - `SHORTEST PATH (a)-[*]->(b) WEIGHTED BY cost` - weighted by property
/// - `SHORTEST PATH (a)-[*]->(b) WEIGHTED BY cost DEFAULT 1.0` - with default
/// - `ALL SHORTEST PATHS (a)-[*]->(b)` - find all shortest paths
///
/// # Returns
///
/// A tuple of (ShortestPathPattern, remaining input).
pub fn parse_shortest_path(input: &str) -> ParseResult<(ShortestPathPattern, &str)> {
    let input = input.trim();
    let input_upper = input.to_uppercase();

    // Check for ALL SHORTEST PATHS or SHORTEST PATH
    let (find_all, remaining) = if input_upper.starts_with("ALL SHORTEST PATHS") {
        (true, input[18..].trim_start())
    } else if input_upper.starts_with("ALL SHORTEST PATH") {
        // Also accept without the S
        (true, input[17..].trim_start())
    } else if input_upper.starts_with("SHORTEST PATHS") {
        (true, input[14..].trim_start())
    } else if input_upper.starts_with("SHORTEST PATH") {
        (false, input[13..].trim_start())
    } else {
        return Err(ParseError::InvalidPattern(
            "expected SHORTEST PATH or ALL SHORTEST PATHS".to_string(),
        ));
    };

    // Parse the path pattern
    let (path, remaining) = ExtendedParser::parse_path_pattern(remaining)?;

    // Check for WEIGHTED BY clause
    let remaining = remaining.trim();
    let remaining_upper = remaining.to_uppercase();

    let (weight, remaining) = if remaining_upper.starts_with("WEIGHTED BY") {
        let after_weighted = remaining[11..].trim_start();
        let (weight_spec, rest) = parse_weight_spec(after_weighted)?;
        (Some(weight_spec), rest)
    } else {
        (None, remaining)
    };

    let pattern = ShortestPathPattern { path, find_all, weight };

    Ok((pattern, remaining))
}

/// Parses a weight specification.
///
/// Handles:
/// - Property name: `cost`
/// - Property with default: `cost DEFAULT 1.0`
/// - Numeric constant: `1.5`
fn parse_weight_spec(input: &str) -> ParseResult<(WeightSpec, &str)> {
    let input = input.trim();

    // Try to parse as a number first
    let num_end =
        input.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').unwrap_or(input.len());

    if num_end > 0 {
        let potential_num = &input[..num_end];
        if let Ok(value) = potential_num.parse::<f64>() {
            // Check if this looks like a number (not an identifier starting with digits)
            if potential_num.chars().next().is_some_and(|c| c.is_ascii_digit() || c == '-') {
                return Ok((WeightSpec::Constant(value), &input[num_end..]));
            }
        }
    }

    // Parse as identifier (property name)
    let ident_end = input.find(|c: char| !c.is_alphanumeric() && c != '_').unwrap_or(input.len());

    if ident_end == 0 {
        return Err(ParseError::InvalidPattern(
            "expected property name or number after WEIGHTED BY".to_string(),
        ));
    }

    let name = input[..ident_end].to_string();
    let remaining = input[ident_end..].trim_start();
    let remaining_upper = remaining.to_uppercase();

    // Check for DEFAULT clause
    let (default, remaining) = if remaining_upper.starts_with("DEFAULT") {
        let after_default = remaining[7..].trim_start();

        // Parse the default value
        let default_end = after_default
            .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
            .unwrap_or(after_default.len());

        if default_end == 0 {
            return Err(ParseError::InvalidPattern("expected number after DEFAULT".to_string()));
        }

        let default_str = &after_default[..default_end];
        let default_value = default_str.parse::<f64>().map_err(|_| {
            ParseError::InvalidPattern(format!("invalid default value: {default_str}"))
        })?;

        (Some(default_value), &after_default[default_end..])
    } else {
        (None, remaining)
    };

    Ok((WeightSpec::Property { name, default }, remaining))
}

/// Parses a vector distance expression.
///
/// This function is used to parse expressions like:
/// - `embedding <-> $query` (Euclidean distance)
/// - `vec_column <=> $param` (Cosine distance)
/// - `data <#> $vector` (Inner product)
pub fn parse_vector_distance(left: Expr, metric: DistanceMetric, right: Expr) -> Expr {
    Expr::BinaryOp {
        left: Box::new(left),
        op: match metric {
            DistanceMetric::Cosine => BinaryOp::CosineDistance,
            DistanceMetric::InnerProduct => BinaryOp::InnerProduct,
            // Euclidean, Manhattan, and Hamming all map to EuclideanDistance (as fallback)
            DistanceMetric::Euclidean | DistanceMetric::Manhattan | DistanceMetric::Hamming => {
                BinaryOp::EuclideanDistance
            }
        },
        right: Box::new(right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::HybridCombinationMethod;

    #[test]
    fn parse_simple_node_pattern() {
        let (node, remaining) = ExtendedParser::parse_node_pattern("(p)").unwrap();
        assert!(remaining.is_empty());
        assert_eq!(node.variable.as_ref().map(|i| i.name.as_str()), Some("p"));
        assert!(node.labels.is_empty());
    }

    #[test]
    fn parse_node_with_label() {
        let (node, _) = ExtendedParser::parse_node_pattern("(p:Person)").unwrap();
        assert_eq!(node.variable.as_ref().map(|i| i.name.as_str()), Some("p"));
        assert_eq!(node.labels.len(), 1);
        assert_eq!(node.labels[0].name, "Person");
    }

    #[test]
    fn parse_node_with_multiple_labels() {
        let (node, _) = ExtendedParser::parse_node_pattern("(p:Person:Employee)").unwrap();
        assert_eq!(node.labels.len(), 2);
        assert_eq!(node.labels[0].name, "Person");
        assert_eq!(node.labels[1].name, "Employee");
    }

    #[test]
    fn parse_anonymous_node() {
        let (node, _) = ExtendedParser::parse_node_pattern("()").unwrap();
        assert!(node.variable.is_none());
        assert!(node.labels.is_empty());
    }

    #[test]
    fn parse_directed_edge() {
        let (edge, remaining) = ExtendedParser::parse_edge_pattern("-[:FOLLOWS]->").unwrap();
        assert!(remaining.is_empty());
        assert_eq!(edge.direction, EdgeDirection::Right);
        assert_eq!(edge.edge_types.len(), 1);
        assert_eq!(edge.edge_types[0].name, "FOLLOWS");
    }

    #[test]
    fn parse_left_edge() {
        let (edge, _) = ExtendedParser::parse_edge_pattern("<-[:CREATED_BY]-").unwrap();
        assert_eq!(edge.direction, EdgeDirection::Left);
        assert_eq!(edge.edge_types[0].name, "CREATED_BY");
    }

    #[test]
    fn parse_undirected_edge() {
        let (edge, _) = ExtendedParser::parse_edge_pattern("-[:KNOWS]-").unwrap();
        assert_eq!(edge.direction, EdgeDirection::Undirected);
    }

    #[test]
    fn parse_edge_with_variable() {
        let (edge, _) = ExtendedParser::parse_edge_pattern("-[r:FOLLOWS]->").unwrap();
        assert_eq!(edge.variable.as_ref().map(|i| i.name.as_str()), Some("r"));
    }

    #[test]
    fn parse_edge_with_length() {
        let (edge, _) = ExtendedParser::parse_edge_pattern("-[:FOLLOWS*1..3]->").unwrap();
        assert_eq!(edge.length, EdgeLength::Range { min: Some(1), max: Some(3) });
    }

    #[test]
    fn parse_edge_any_length() {
        let (edge, _) = ExtendedParser::parse_edge_pattern("-[:PATH*]->").unwrap();
        assert_eq!(edge.length, EdgeLength::Any);
    }

    #[test]
    fn parse_simple_path() {
        let (path, _) = ExtendedParser::parse_path_pattern("(a)-[:FOLLOWS]->(b)").unwrap();
        assert_eq!(path.start.variable.as_ref().map(|i| i.name.as_str()), Some("a"));
        assert_eq!(path.steps.len(), 1);
    }

    #[test]
    fn parse_long_path() {
        let (path, _) =
            ExtendedParser::parse_path_pattern("(a)-[:KNOWS]->(b)-[:LIKES]->(c)").unwrap();
        assert_eq!(path.steps.len(), 2);
    }

    #[test]
    fn parse_graph_pattern() {
        let pattern = ExtendedParser::parse_graph_pattern("(u:User)-[:FOLLOWS]->(f:User)").unwrap();
        assert_eq!(pattern.paths.len(), 1);
    }

    #[test]
    fn parse_multiple_paths() {
        let pattern =
            ExtendedParser::parse_graph_pattern("(a)-[:R1]->(b), (b)-[:R2]->(c)").unwrap();
        assert_eq!(pattern.paths.len(), 2);
    }

    #[test]
    fn extract_match_clause() {
        let (sql, patterns, optional_patterns) = ExtendedParser::extract_match_clauses(
            "SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1",
        )
        .unwrap();

        assert!(sql.contains("SELECT * FROM users"));
        assert!(sql.contains("WHERE u.id = 1"));
        assert!(!sql.to_uppercase().contains("MATCH"));
        assert_eq!(patterns.len(), 1);
        // No optional patterns in this query
        assert!(optional_patterns.is_empty() || optional_patterns.iter().all(|v| v.is_empty()));
    }

    #[test]
    fn parse_extended_select() {
        let stmts =
            ExtendedParser::parse("SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1")
                .unwrap();

        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            assert!(select.match_clause.is_some());
            assert!(select.where_clause.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn preprocess_vector_ops() {
        let result = ExtendedParser::preprocess_vector_ops("a <-> b");
        assert!(result.contains("__VEC_EUCLIDEAN__"));

        let result = ExtendedParser::preprocess_vector_ops("a <=> b");
        assert!(result.contains("__VEC_COSINE__"));

        let result = ExtendedParser::preprocess_vector_ops("a <#> b");
        assert!(result.contains("__VEC_INNER__"));

        let result = ExtendedParser::preprocess_vector_ops("a <##> b");
        assert!(result.contains("__VEC_MAXSIM__"));
    }

    #[test]
    fn preprocess_maxsim_before_inner() {
        // Ensure <##> is processed before <#> to avoid incorrect matching
        let result = ExtendedParser::preprocess_vector_ops("a <##> b");
        assert!(result.contains("__VEC_MAXSIM__"));
        assert!(!result.contains("__VEC_INNER__"));
    }

    #[test]
    fn parse_node_with_properties() {
        let (node, _) =
            ExtendedParser::parse_node_pattern("(p:Person {name: 'Alice', age: 30})").unwrap();
        assert_eq!(node.properties.len(), 2);
        assert_eq!(node.properties[0].name.name, "name");
        assert_eq!(node.properties[1].name.name, "age");
    }

    #[test]
    fn parse_edge_length_exact() {
        let length = ExtendedParser::parse_edge_length("3").unwrap();
        assert_eq!(length, EdgeLength::Exact(3));
    }

    #[test]
    fn parse_edge_length_range() {
        let length = ExtendedParser::parse_edge_length("1..5").unwrap();
        assert_eq!(length, EdgeLength::Range { min: Some(1), max: Some(5) });
    }

    #[test]
    fn parse_edge_length_min_only() {
        let length = ExtendedParser::parse_edge_length("2..").unwrap();
        assert_eq!(length, EdgeLength::Range { min: Some(2), max: None });
    }

    #[test]
    fn parse_edge_length_max_only() {
        let length = ExtendedParser::parse_edge_length("..5").unwrap();
        assert_eq!(length, EdgeLength::Range { min: None, max: Some(5) });
    }

    #[test]
    fn parse_shortest_path_unweighted() {
        let (sp, remaining) = parse_shortest_path("SHORTEST PATH (a)-[*]->(b)").unwrap();
        assert!(!sp.find_all);
        assert!(sp.weight.is_none());
        assert!(remaining.is_empty());
    }

    #[test]
    fn parse_shortest_path_weighted() {
        let (sp, _) = parse_shortest_path("SHORTEST PATH (a)-[*]->(b) WEIGHTED BY cost").unwrap();
        assert!(!sp.find_all);
        assert!(sp.weight.is_some());
        match sp.weight.unwrap() {
            WeightSpec::Property { name, default } => {
                assert_eq!(name, "cost");
                assert!(default.is_none());
            }
            _ => panic!("expected Property weight spec"),
        }
    }

    #[test]
    fn parse_shortest_path_weighted_with_default() {
        let (sp, _) =
            parse_shortest_path("SHORTEST PATH (a)-[*]->(b) WEIGHTED BY distance DEFAULT 1.0")
                .unwrap();
        match sp.weight.unwrap() {
            WeightSpec::Property { name, default } => {
                assert_eq!(name, "distance");
                assert_eq!(default, Some(1.0));
            }
            _ => panic!("expected Property weight spec"),
        }
    }

    #[test]
    fn parse_all_shortest_paths() {
        let (sp, _) = parse_shortest_path("ALL SHORTEST PATHS (a)-[*]->(b)").unwrap();
        assert!(sp.find_all);
        assert!(sp.weight.is_none());
    }

    #[test]
    fn parse_shortest_path_constant_weight() {
        let (sp, _) = parse_shortest_path("SHORTEST PATH (a)-[*]->(b) WEIGHTED BY 2.5").unwrap();
        match sp.weight.unwrap() {
            WeightSpec::Constant(v) => assert_eq!(v, 2.5),
            _ => panic!("expected Constant weight spec"),
        }
    }

    #[test]
    fn parse_shortest_path_with_edge_type() {
        let (sp, _) =
            parse_shortest_path("SHORTEST PATH (a)-[:ROAD*]->(b) WEIGHTED BY distance").unwrap();
        assert_eq!(sp.path.steps.len(), 1);
        let (edge, _) = &sp.path.steps[0];
        assert_eq!(edge.edge_types.len(), 1);
        assert_eq!(edge.edge_types[0].name, "ROAD");
    }

    #[test]
    fn parse_hybrid_function_basic() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM docs ORDER BY HYBRID(dense <=> $1, 0.7, sparse <#> $2, 0.3) LIMIT 10",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            assert_eq!(select.order_by.len(), 1);
            // The ORDER BY should contain a HybridSearch expression
            let order_expr = &*select.order_by[0].expr;
            assert!(matches!(order_expr, Expr::HybridSearch { .. }));
            if let Expr::HybridSearch { components, method } = order_expr {
                assert_eq!(components.len(), 2);
                assert!((components[0].weight - 0.7).abs() < 0.001);
                assert!((components[1].weight - 0.3).abs() < 0.001);
                assert!(matches!(method, HybridCombinationMethod::WeightedSum));
            }
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn parse_rrf_function() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM docs ORDER BY RRF(dense <=> $1, sparse <#> $2) LIMIT 10",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            let order_expr = &*select.order_by[0].expr;
            if let Expr::HybridSearch { components, method } = order_expr {
                assert_eq!(components.len(), 2);
                // RRF uses equal weights (1.0)
                assert!((components[0].weight - 1.0).abs() < 0.001);
                assert!((components[1].weight - 1.0).abs() < 0.001);
                assert!(matches!(method, HybridCombinationMethod::RRF { k: 60 }));
            } else {
                panic!("Expected HybridSearch expression");
            }
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn parse_hybrid_function_preserves_vector_ops() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM docs ORDER BY HYBRID(embedding <=> $q1, 0.5, sparse <#> $q2, 0.5)",
        )
        .unwrap();
        if let Statement::Select(select) = &stmts[0] {
            if let Expr::HybridSearch { components, .. } = &*select.order_by[0].expr {
                // First component should be cosine distance
                if let Expr::BinaryOp { op: BinaryOp::CosineDistance, .. } =
                    components[0].distance_expr.as_ref()
                {
                    // OK
                } else {
                    panic!("Expected CosineDistance operator for first component");
                }
                // Second component should be inner product
                if let Expr::BinaryOp { op: BinaryOp::InnerProduct, .. } =
                    components[1].distance_expr.as_ref()
                {
                    // OK
                } else {
                    panic!("Expected InnerProduct operator for second component");
                }
            } else {
                panic!("Expected HybridSearch expression");
            }
        }
    }

    // CREATE COLLECTION tests

    #[test]
    fn parse_create_collection_basic() {
        let stmts = ExtendedParser::parse(
            "CREATE COLLECTION documents (dense VECTOR(768) USING hnsw WITH (distance = 'cosine'))",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::CreateCollection(create) = &stmts[0] {
            assert_eq!(create.name.name, "documents");
            assert!(!create.if_not_exists);
            assert_eq!(create.vectors.len(), 1);
            assert_eq!(create.vectors[0].name.name, "dense");
            assert!(matches!(
                create.vectors[0].vector_type,
                VectorTypeDef::Vector { dimension: 768 }
            ));
            assert_eq!(create.vectors[0].using, Some("hnsw".to_string()));
            assert_eq!(create.vectors[0].with_options.len(), 1);
            assert_eq!(
                create.vectors[0].with_options[0],
                ("distance".to_string(), "cosine".to_string())
            );
        } else {
            panic!("Expected CreateCollection statement");
        }
    }

    #[test]
    fn parse_create_collection_if_not_exists() {
        let stmts =
            ExtendedParser::parse("CREATE IF NOT EXISTS COLLECTION docs (v VECTOR(128))").unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::CreateCollection(create) = &stmts[0] {
            assert!(create.if_not_exists);
            assert_eq!(create.name.name, "docs");
        } else {
            panic!("Expected CreateCollection statement");
        }
    }

    #[test]
    fn parse_create_collection_multiple_vectors() {
        let stmts = ExtendedParser::parse(
            "CREATE COLLECTION documents (
                dense VECTOR(768) USING hnsw WITH (distance = 'cosine'),
                sparse SPARSE_VECTOR USING inverted,
                colbert MULTI_VECTOR(128) USING hnsw WITH (aggregation = 'maxsim')
            )",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::CreateCollection(create) = &stmts[0] {
            assert_eq!(create.vectors.len(), 3);

            // Check dense vector
            assert_eq!(create.vectors[0].name.name, "dense");
            assert!(matches!(
                create.vectors[0].vector_type,
                VectorTypeDef::Vector { dimension: 768 }
            ));

            // Check sparse vector
            assert_eq!(create.vectors[1].name.name, "sparse");
            assert!(matches!(
                create.vectors[1].vector_type,
                VectorTypeDef::SparseVector { max_dimension: None }
            ));
            assert_eq!(create.vectors[1].using, Some("inverted".to_string()));

            // Check multi-vector
            assert_eq!(create.vectors[2].name.name, "colbert");
            assert!(matches!(
                create.vectors[2].vector_type,
                VectorTypeDef::MultiVector { token_dim: 128 }
            ));
        } else {
            panic!("Expected CreateCollection statement");
        }
    }

    #[test]
    fn parse_create_collection_sparse_with_max_dim() {
        let stmts = ExtendedParser::parse(
            "CREATE COLLECTION docs (keywords SPARSE_VECTOR(30522) USING inverted)",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::CreateCollection(create) = &stmts[0] {
            assert!(matches!(
                create.vectors[0].vector_type,
                VectorTypeDef::SparseVector { max_dimension: Some(30522) }
            ));
        } else {
            panic!("Expected CreateCollection statement");
        }
    }

    #[test]
    fn parse_create_collection_binary_vector() {
        let stmts =
            ExtendedParser::parse("CREATE COLLECTION docs (hash BINARY_VECTOR(1024))").unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::CreateCollection(create) = &stmts[0] {
            assert!(matches!(
                create.vectors[0].vector_type,
                VectorTypeDef::BinaryVector { bits: 1024 }
            ));
        } else {
            panic!("Expected CreateCollection statement");
        }
    }

    #[test]
    fn parse_create_collection_multiple_with_options() {
        let stmts = ExtendedParser::parse(
            "CREATE COLLECTION docs (vec VECTOR(768) USING hnsw WITH (distance = 'euclidean', m = 16, ef_construction = 200))",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::CreateCollection(create) = &stmts[0] {
            assert_eq!(create.vectors[0].with_options.len(), 3);
            assert!(create.vectors[0]
                .with_options
                .contains(&("distance".to_string(), "euclidean".to_string())));
            assert!(create.vectors[0].with_options.contains(&("m".to_string(), "16".to_string())));
            assert!(create.vectors[0]
                .with_options
                .contains(&("ef_construction".to_string(), "200".to_string())));
        } else {
            panic!("Expected CreateCollection statement");
        }
    }

    #[test]
    fn parse_create_collection_flat_index() {
        let stmts =
            ExtendedParser::parse("CREATE COLLECTION docs (vec VECTOR(768) USING flat)").unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::CreateCollection(create) = &stmts[0] {
            assert_eq!(create.vectors[0].using, Some("flat".to_string()));
        } else {
            panic!("Expected CreateCollection statement");
        }
    }

    #[test]
    fn parse_create_collection_no_using() {
        let stmts = ExtendedParser::parse("CREATE COLLECTION docs (vec VECTOR(768))").unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::CreateCollection(create) = &stmts[0] {
            assert!(create.vectors[0].using.is_none());
        } else {
            panic!("Expected CreateCollection statement");
        }
    }

    #[test]
    fn is_create_collection_detection() {
        assert!(ExtendedParser::is_create_collection("CREATE COLLECTION foo (v VECTOR(10))"));
        assert!(ExtendedParser::is_create_collection("  CREATE COLLECTION foo (v VECTOR(10))  "));
        assert!(ExtendedParser::is_create_collection(
            "CREATE IF NOT EXISTS COLLECTION foo (v VECTOR(10))"
        ));
        assert!(!ExtendedParser::is_create_collection("CREATE TABLE foo (id INT)"));
        assert!(!ExtendedParser::is_create_collection("SELECT * FROM foo"));
    }

    // New syntax tests for named embeddings

    #[test]
    fn parse_create_collection_new_syntax() {
        let stmts = ExtendedParser::parse(
            "CREATE COLLECTION documents (
                title TEXT,
                content TEXT,
                VECTOR text_embedding DIMENSION 1536,
                VECTOR image_embedding DIMENSION 512
            )",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::CreateCollection(create) = &stmts[0] {
            assert_eq!(create.name.name, "documents");
            // Check payload fields
            assert_eq!(create.payload_fields.len(), 2);
            assert_eq!(create.payload_fields[0].name.name, "title");
            assert!(matches!(create.payload_fields[0].data_type, DataType::Text));
            assert_eq!(create.payload_fields[1].name.name, "content");
            assert!(matches!(create.payload_fields[1].data_type, DataType::Text));
            // Check vectors
            assert_eq!(create.vectors.len(), 2);
            assert_eq!(create.vectors[0].name.name, "text_embedding");
            assert!(matches!(
                create.vectors[0].vector_type,
                VectorTypeDef::Vector { dimension: 1536 }
            ));
            assert_eq!(create.vectors[1].name.name, "image_embedding");
            assert!(matches!(
                create.vectors[1].vector_type,
                VectorTypeDef::Vector { dimension: 512 }
            ));
        } else {
            panic!("Expected CreateCollection statement");
        }
    }

    #[test]
    fn parse_create_collection_mixed_syntax() {
        let stmts = ExtendedParser::parse(
            "CREATE COLLECTION documents (
                title TEXT,
                category INTEGER INDEXED,
                dense VECTOR(768) USING hnsw,
                VECTOR summary_embedding DIMENSION 1536 USING hnsw WITH (distance = 'cosine')
            )",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::CreateCollection(create) = &stmts[0] {
            // Check payload fields
            assert_eq!(create.payload_fields.len(), 2);
            assert_eq!(create.payload_fields[0].name.name, "title");
            assert!(!create.payload_fields[0].indexed);
            assert_eq!(create.payload_fields[1].name.name, "category");
            assert!(create.payload_fields[1].indexed);
            // Check vectors (both legacy and new syntax)
            assert_eq!(create.vectors.len(), 2);
            assert_eq!(create.vectors[0].name.name, "dense");
            assert_eq!(create.vectors[1].name.name, "summary_embedding");
            assert!(matches!(
                create.vectors[1].vector_type,
                VectorTypeDef::Vector { dimension: 1536 }
            ));
            assert_eq!(create.vectors[1].using, Some("hnsw".to_string()));
        } else {
            panic!("Expected CreateCollection statement");
        }
    }

    #[test]
    fn parse_create_collection_various_types() {
        let stmts = ExtendedParser::parse(
            "CREATE COLLECTION items (
                name VARCHAR(255),
                count INTEGER,
                price FLOAT,
                active BOOLEAN,
                metadata JSON,
                created_at TIMESTAMP,
                id UUID,
                VECTOR embedding DIMENSION 768
            )",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::CreateCollection(create) = &stmts[0] {
            assert_eq!(create.payload_fields.len(), 7);
            assert!(matches!(create.payload_fields[0].data_type, DataType::Varchar(Some(255))));
            assert!(matches!(create.payload_fields[1].data_type, DataType::Integer));
            assert!(matches!(create.payload_fields[2].data_type, DataType::Real));
            assert!(matches!(create.payload_fields[3].data_type, DataType::Boolean));
            assert!(matches!(create.payload_fields[4].data_type, DataType::Json));
            assert!(matches!(create.payload_fields[5].data_type, DataType::Timestamp));
            assert!(matches!(create.payload_fields[6].data_type, DataType::Uuid));
            assert_eq!(create.vectors.len(), 1);
        } else {
            panic!("Expected CreateCollection statement");
        }
    }

    // Similarity function tests

    #[test]
    fn parse_cosine_similarity_function() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM documents ORDER BY COSINE_SIMILARITY(text_embedding, $query) LIMIT 10",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            assert_eq!(select.order_by.len(), 1);
            if let Expr::BinaryOp { op: BinaryOp::CosineDistance, left, right } =
                &*select.order_by[0].expr
            {
                // Check left side is the vector column reference
                if let Expr::Column(col) = left.as_ref() {
                    assert_eq!(col.name().map(|id| id.name.as_str()), Some("text_embedding"));
                } else {
                    panic!("Expected column reference for left operand");
                }
                // Check right side is a parameter
                assert!(matches!(right.as_ref(), Expr::Parameter(_)));
            } else {
                panic!("Expected BinaryOp with CosineDistance");
            }
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn parse_euclidean_distance_function() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM docs ORDER BY EUCLIDEAN_DISTANCE(embedding, $1) ASC LIMIT 5",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            if let Expr::BinaryOp { op: BinaryOp::EuclideanDistance, .. } =
                &*select.order_by[0].expr
            {
                // OK - correctly parsed
            } else {
                panic!("Expected BinaryOp with EuclideanDistance");
            }
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn parse_l2_distance_alias() {
        let stmts =
            ExtendedParser::parse("SELECT * FROM docs ORDER BY L2_DISTANCE(vec, $q) LIMIT 10")
                .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            assert!(matches!(
                &*select.order_by[0].expr,
                Expr::BinaryOp { op: BinaryOp::EuclideanDistance, .. }
            ));
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn parse_inner_product_function() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM docs ORDER BY INNER_PRODUCT(vec, $q) DESC LIMIT 10",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            assert!(matches!(
                &*select.order_by[0].expr,
                Expr::BinaryOp { op: BinaryOp::InnerProduct, .. }
            ));
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn parse_dot_product_alias() {
        let stmts =
            ExtendedParser::parse("SELECT * FROM docs ORDER BY DOT_PRODUCT(vec, $q) DESC LIMIT 10")
                .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            assert!(matches!(
                &*select.order_by[0].expr,
                Expr::BinaryOp { op: BinaryOp::InnerProduct, .. }
            ));
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn parse_maxsim_function() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM docs ORDER BY MAXSIM(colbert_vec, $q) DESC LIMIT 10",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            assert!(matches!(
                &*select.order_by[0].expr,
                Expr::BinaryOp { op: BinaryOp::MaxSim, .. }
            ));
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn parse_distance_function_case_insensitive() {
        // Test that function names are case-insensitive
        let stmts = ExtendedParser::parse(
            "SELECT * FROM docs ORDER BY cosine_similarity(vec, $q) LIMIT 10",
        )
        .unwrap();
        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            assert!(matches!(
                &*select.order_by[0].expr,
                Expr::BinaryOp { op: BinaryOp::CosineDistance, .. }
            ));
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn parse_distance_function_name_mapping() {
        // Test the function name mapping directly
        assert!(matches!(
            ExtendedParser::parse_distance_function_name("COSINE_SIMILARITY"),
            Some(BinaryOp::CosineDistance)
        ));
        assert!(matches!(
            ExtendedParser::parse_distance_function_name("COS_DISTANCE"),
            Some(BinaryOp::CosineDistance)
        ));
        assert!(matches!(
            ExtendedParser::parse_distance_function_name("EUCLIDEAN_DISTANCE"),
            Some(BinaryOp::EuclideanDistance)
        ));
        assert!(matches!(
            ExtendedParser::parse_distance_function_name("L2"),
            Some(BinaryOp::EuclideanDistance)
        ));
        assert!(matches!(
            ExtendedParser::parse_distance_function_name("INNER_PRODUCT"),
            Some(BinaryOp::InnerProduct)
        ));
        assert!(matches!(
            ExtendedParser::parse_distance_function_name("DOT"),
            Some(BinaryOp::InnerProduct)
        ));
        assert!(matches!(
            ExtendedParser::parse_distance_function_name("MAXSIM"),
            Some(BinaryOp::MaxSim)
        ));
        assert!(ExtendedParser::parse_distance_function_name("UNKNOWN").is_none());
    }

    // ============================================================================
    // OPTIONAL MATCH Tests
    // ============================================================================

    #[test]
    fn extract_optional_match_clause() {
        let (sql, patterns, optional_patterns) = ExtendedParser::extract_match_clauses(
            "SELECT u.name, p.title \
             FROM users \
             MATCH (u:User) \
             OPTIONAL MATCH (u)-[:LIKES]->(p:Post) \
             WHERE u.status = 'active'",
        )
        .unwrap();

        // The SQL should not contain MATCH or OPTIONAL MATCH
        assert!(!sql.to_uppercase().contains("MATCH"));
        assert!(!sql.to_uppercase().contains("OPTIONAL"));

        // One required MATCH pattern
        assert_eq!(patterns.len(), 1);

        // One set of optional patterns with one pattern in it
        assert_eq!(optional_patterns.len(), 1);
        assert_eq!(optional_patterns[0].len(), 1);
    }

    #[test]
    fn extract_multiple_optional_match_clauses() {
        let (sql, patterns, optional_patterns) = ExtendedParser::extract_match_clauses(
            "SELECT u.name, p.title, c.text \
             FROM entities \
             MATCH (u:User) \
             OPTIONAL MATCH (u)-[:LIKES]->(p:Post) \
             OPTIONAL MATCH (u)-[:WROTE]->(c:Comment) \
             WHERE u.active = true",
        )
        .unwrap();

        assert!(!sql.to_uppercase().contains("MATCH"));
        assert_eq!(patterns.len(), 1);
        assert_eq!(optional_patterns.len(), 1);
        // Two optional patterns for the same statement
        assert_eq!(optional_patterns[0].len(), 2);
    }

    #[test]
    fn parse_optional_match_in_select() {
        let stmts = ExtendedParser::parse(
            "SELECT u.name, p.title \
             FROM users \
             MATCH (u:User) \
             OPTIONAL MATCH (u)-[:LIKES]->(p:Post) \
             WHERE u.status = 'active'",
        )
        .unwrap();

        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            // Required MATCH should be present
            assert!(select.match_clause.is_some());
            // One optional MATCH clause
            assert_eq!(select.optional_match_clauses.len(), 1);
            // WHERE should be preserved
            assert!(select.where_clause.is_some());
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn parse_optional_match_pattern_structure() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM entities MATCH (u:User) OPTIONAL MATCH (u)-[:FOLLOWS]->(f:User)",
        )
        .unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let optional = &select.optional_match_clauses[0];
            // The optional pattern should have one path
            assert_eq!(optional.paths.len(), 1);
            let path = &optional.paths[0];
            // Start node should be `u`
            assert_eq!(path.start.variable.as_ref().map(|v| v.name.as_str()), Some("u"));
            // Should have one step
            assert_eq!(path.steps.len(), 1);
            // Edge type should be FOLLOWS
            let (edge, node) = &path.steps[0];
            assert_eq!(edge.edge_types[0].name, "FOLLOWS");
            // End node should be `f:User`
            assert_eq!(node.variable.as_ref().map(|v| v.name.as_str()), Some("f"));
            assert_eq!(node.labels[0].name, "User");
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn find_optional_match_keyword() {
        // Should find OPTIONAL MATCH at position 0
        assert_eq!(ExtendedParser::find_optional_match_keyword("OPTIONAL MATCH (a)"), Some(0));

        // Should find with leading whitespace
        assert_eq!(ExtendedParser::find_optional_match_keyword("  OPTIONAL MATCH (a)"), Some(2));

        // Should not find plain MATCH
        assert_eq!(ExtendedParser::find_optional_match_keyword("MATCH (a)"), None);

        // Should not find OPTIONAL without MATCH
        assert_eq!(ExtendedParser::find_optional_match_keyword("OPTIONAL something else"), None);

        // Case insensitive
        assert_eq!(ExtendedParser::find_optional_match_keyword("optional match (a)"), Some(0));
    }

    #[test]
    fn find_match_skips_optional_match() {
        // find_match_keyword should NOT match the MATCH inside "OPTIONAL MATCH"
        let input = "OPTIONAL MATCH (a) MATCH (b)";
        let pos = ExtendedParser::find_match_keyword(input);
        // Should find the standalone MATCH, not the one in OPTIONAL MATCH
        assert_eq!(pos, Some(19)); // Position of the second MATCH

        // When there's only OPTIONAL MATCH, should return None
        assert_eq!(ExtendedParser::find_match_keyword("OPTIONAL MATCH (a)"), None);
    }

    #[test]
    fn optional_match_order_of_clauses() {
        // OPTIONAL MATCH should come after required MATCH
        let (_, patterns, optional_patterns) = ExtendedParser::extract_match_clauses(
            "SELECT * FROM entities MATCH (a) OPTIONAL MATCH (b)",
        )
        .unwrap();

        assert_eq!(patterns.len(), 1);
        assert_eq!(optional_patterns.len(), 1);
        assert_eq!(optional_patterns[0].len(), 1);
    }

    // ========== CREATE Tests ==========

    #[test]
    fn is_cypher_create_basic() {
        assert!(ExtendedParser::is_cypher_create("CREATE (n:Person)"));
        assert!(ExtendedParser::is_cypher_create("create (n)"));
        assert!(ExtendedParser::is_cypher_create("  CREATE (n:Label {name: 'test'})"));
    }

    #[test]
    fn is_cypher_create_false_for_sql() {
        // SQL CREATE should not be detected as Cypher CREATE
        assert!(!ExtendedParser::is_cypher_create("CREATE TABLE users (id INT)"));
        assert!(!ExtendedParser::is_cypher_create("CREATE INDEX idx ON users(id)"));
    }

    #[test]
    fn parse_cypher_create_simple_node() {
        let result = ExtendedParser::parse_cypher_create("CREATE (n)");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Create(stmt) => {
                assert!(stmt.match_clause.is_none());
                assert!(stmt.where_clause.is_none());
                assert_eq!(stmt.patterns.len(), 1);
            }
            _ => panic!("Expected CreateGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_create_node_with_label() {
        let result = ExtendedParser::parse_cypher_create("CREATE (n:Person)");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Create(stmt) => {
                assert_eq!(stmt.patterns.len(), 1);
                match &stmt.patterns[0] {
                    CreatePattern::Node { variable, labels, .. } => {
                        assert_eq!(variable.as_ref().map(|i| i.name.as_str()), Some("n"));
                        assert_eq!(labels.len(), 1);
                        assert_eq!(labels[0].name, "Person");
                    }
                    _ => panic!("Expected node pattern"),
                }
            }
            _ => panic!("Expected CreateGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_create_node_with_properties() {
        let result =
            ExtendedParser::parse_cypher_create("CREATE (n:Person {name: 'Alice', age: 30})");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Create(stmt) => {
                assert_eq!(stmt.patterns.len(), 1);
                match &stmt.patterns[0] {
                    CreatePattern::Node { properties, .. } => {
                        assert_eq!(properties.len(), 2);
                        assert_eq!(properties[0].0.name, "name");
                        assert_eq!(properties[1].0.name, "age");
                    }
                    _ => panic!("Expected node pattern"),
                }
            }
            _ => panic!("Expected CreateGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_create_relationship() {
        let result = ExtendedParser::parse_cypher_create("CREATE (a)-[:KNOWS]->(b)");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Create(stmt) => {
                assert_eq!(stmt.patterns.len(), 1);
                match &stmt.patterns[0] {
                    CreatePattern::Path { start, steps } => {
                        match start {
                            CreateNodeRef::New { variable, .. } => {
                                assert_eq!(variable.as_ref().map(|i| i.name.as_str()), Some("a"));
                            }
                            CreateNodeRef::Variable(ident) => {
                                assert_eq!(ident.name.as_str(), "a");
                            }
                        }
                        assert_eq!(steps.len(), 1);
                        assert_eq!(steps[0].rel_type.name, "KNOWS");
                    }
                    _ => panic!("Expected path pattern"),
                }
            }
            _ => panic!("Expected Create statement"),
        }
    }

    #[test]
    fn parse_cypher_create_with_match() {
        let result = ExtendedParser::parse_cypher_create(
            "MATCH (a:Person {name: 'Alice'}) CREATE (b:Person)-[:FRIEND]->(a)",
        );
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Create(stmt) => {
                assert!(stmt.match_clause.is_some());
                assert_eq!(stmt.patterns.len(), 1);
            }
            _ => panic!("Expected CreateGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_create_with_return() {
        let result = ExtendedParser::parse_cypher_create("CREATE (n:Person) RETURN n");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Create(stmt) => {
                assert_eq!(stmt.return_clause.len(), 1);
            }
            _ => panic!("Expected CreateGraph statement"),
        }
    }

    // ========== MERGE Tests ==========

    #[test]
    fn is_cypher_merge_basic() {
        assert!(ExtendedParser::is_cypher_merge("MERGE (n:Person)"));
        assert!(ExtendedParser::is_cypher_merge("merge (n)"));
        assert!(ExtendedParser::is_cypher_merge("  MERGE (n:Label {id: 1})"));
    }

    #[test]
    fn is_cypher_merge_false_for_non_merge() {
        assert!(!ExtendedParser::is_cypher_merge("CREATE (n:Person)"));
        assert!(!ExtendedParser::is_cypher_merge("SELECT * FROM table"));
    }

    #[test]
    fn parse_cypher_merge_simple_node() {
        let result = ExtendedParser::parse_cypher_merge("MERGE (n:Person)");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Merge(stmt) => {
                assert!(stmt.match_clause.is_none());
                assert!(stmt.where_clause.is_none());
                assert!(stmt.on_create.is_empty());
                assert!(stmt.on_match.is_empty());
            }
            _ => panic!("Expected MergeGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_merge_with_key_properties() {
        let result =
            ExtendedParser::parse_cypher_merge("MERGE (n:Person {email: 'test@example.com'})");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Merge(stmt) => match &stmt.pattern {
                MergePattern::Node { match_properties, .. } => {
                    assert_eq!(match_properties.len(), 1);
                    assert_eq!(match_properties[0].0.name, "email");
                }
                _ => panic!("Expected node pattern"),
            },
            _ => panic!("Expected MergeGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_merge_with_on_create_set() {
        let result = ExtendedParser::parse_cypher_merge(
            "MERGE (n:Person {id: 1}) ON CREATE SET n.created = true",
        );
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Merge(stmt) => {
                assert_eq!(stmt.on_create.len(), 1);
                assert!(stmt.on_match.is_empty());
                match &stmt.on_create[0] {
                    SetAction::Property { variable, property, .. } => {
                        assert_eq!(variable.name, "n");
                        assert_eq!(property.name, "created");
                    }
                    _ => panic!("Expected Property action"),
                }
            }
            _ => panic!("Expected MergeGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_merge_with_on_match_set() {
        let result = ExtendedParser::parse_cypher_merge(
            "MERGE (n:Person {id: 1}) ON MATCH SET n.updated = true",
        );
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Merge(stmt) => {
                assert!(stmt.on_create.is_empty());
                assert_eq!(stmt.on_match.len(), 1);
                match &stmt.on_match[0] {
                    SetAction::Property { variable, property, .. } => {
                        assert_eq!(variable.name, "n");
                        assert_eq!(property.name, "updated");
                    }
                    _ => panic!("Expected Property action"),
                }
            }
            _ => panic!("Expected MergeGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_merge_with_both_on_clauses() {
        let result = ExtendedParser::parse_cypher_merge(
            "MERGE (n:Person {id: 1}) ON CREATE SET n.created = true ON MATCH SET n.updated = true",
        );
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Merge(stmt) => {
                assert_eq!(stmt.on_create.len(), 1);
                assert_eq!(stmt.on_match.len(), 1);
            }
            _ => panic!("Expected MergeGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_merge_with_return() {
        let result = ExtendedParser::parse_cypher_merge("MERGE (n:Person {id: 1}) RETURN n");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Merge(stmt) => {
                assert_eq!(stmt.return_clause.len(), 1);
            }
            _ => panic!("Expected MergeGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_merge_with_match() {
        let result =
            ExtendedParser::parse_cypher_merge("MATCH (p:Person) MERGE (n:Account {owner: p.id})");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Merge(stmt) => {
                assert!(stmt.match_clause.is_some());
            }
            _ => panic!("Expected MergeGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_merge_relationship() {
        let result = ExtendedParser::parse_cypher_merge("MERGE (a:Person)-[:KNOWS]->(b:Person)");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Merge(stmt) => match &stmt.pattern {
                MergePattern::Relationship { rel_type, .. } => {
                    assert_eq!(rel_type.name, "KNOWS");
                }
                _ => panic!("Expected relationship pattern"),
            },
            _ => panic!("Expected MergeGraph statement"),
        }
    }

    // ========== SET Tests ==========

    #[test]
    fn parse_cypher_set_property() {
        let result = ExtendedParser::parse("MATCH (u:User {name: 'Alice'}) SET u.verified = true");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Set(stmt) => {
                assert_eq!(stmt.set_actions.len(), 1);
                match &stmt.set_actions[0] {
                    SetAction::Property { variable, property, .. } => {
                        assert_eq!(variable.name, "u");
                        assert_eq!(property.name, "verified");
                    }
                    _ => panic!("Expected property assignment"),
                }
            }
            _ => panic!("Expected Set statement"),
        }
    }

    #[test]
    fn parse_cypher_set_multiple_properties() {
        let result = ExtendedParser::parse(
            "MATCH (u:User) WHERE u.name = 'Alice' SET u.verified = true, u.updated = 123",
        );
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Set(stmt) => {
                assert!(stmt.where_clause.is_some());
                assert_eq!(stmt.set_actions.len(), 2);
            }
            _ => panic!("Expected Set statement"),
        }
    }

    #[test]
    fn parse_cypher_set_label() {
        let result = ExtendedParser::parse("MATCH (u:User {name: 'Alice'}) SET u:Admin");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Set(stmt) => {
                assert_eq!(stmt.set_actions.len(), 1);
                match &stmt.set_actions[0] {
                    SetAction::Label { variable, label } => {
                        assert_eq!(variable.name, "u");
                        assert_eq!(label.name, "Admin");
                    }
                    _ => panic!("Expected label assignment"),
                }
            }
            _ => panic!("Expected Set statement"),
        }
    }

    #[test]
    fn parse_cypher_set_with_return() {
        let result = ExtendedParser::parse("MATCH (u:User) SET u.verified = true RETURN u");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Set(stmt) => {
                assert_eq!(stmt.return_clause.len(), 1);
            }
            _ => panic!("Expected Set statement"),
        }
    }

    // ========== DELETE Tests ==========

    #[test]
    fn parse_cypher_delete_node() {
        let result = ExtendedParser::parse("MATCH (u:User {name: 'Alice'}) DELETE u");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::DeleteGraph(stmt) => {
                assert!(!stmt.detach);
                assert_eq!(stmt.variables.len(), 1);
                assert_eq!(stmt.variables[0].name, "u");
            }
            _ => panic!("Expected DeleteGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_detach_delete() {
        let result = ExtendedParser::parse("MATCH (u:User {name: 'Alice'}) DETACH DELETE u");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::DeleteGraph(stmt) => {
                assert!(stmt.detach);
                assert_eq!(stmt.variables.len(), 1);
            }
            _ => panic!("Expected DeleteGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_delete_relationship() {
        let result = ExtendedParser::parse(
            "MATCH (a:User)-[r:FOLLOWS]->(b:User) WHERE a.name = 'Alice' DELETE r",
        );
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::DeleteGraph(stmt) => {
                assert!(!stmt.detach);
                assert_eq!(stmt.variables.len(), 1);
                assert_eq!(stmt.variables[0].name, "r");
                assert!(stmt.where_clause.is_some());
            }
            _ => panic!("Expected DeleteGraph statement"),
        }
    }

    #[test]
    fn parse_cypher_delete_with_return() {
        let result = ExtendedParser::parse("MATCH (u:User) DELETE u RETURN u.name");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::DeleteGraph(stmt) => {
                assert_eq!(stmt.return_clause.len(), 1);
            }
            _ => panic!("Expected DeleteGraph statement"),
        }
    }

    // ========== REMOVE Tests ==========

    #[test]
    fn parse_cypher_remove_property() {
        let result = ExtendedParser::parse("MATCH (u:User {name: 'Alice'}) REMOVE u.temporary");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Remove(stmt) => {
                assert_eq!(stmt.items.len(), 1);
                match &stmt.items[0] {
                    RemoveItem::Property { variable, property } => {
                        assert_eq!(variable.name, "u");
                        assert_eq!(property.name, "temporary");
                    }
                    _ => panic!("Expected property removal"),
                }
            }
            _ => panic!("Expected Remove statement"),
        }
    }

    #[test]
    fn parse_cypher_remove_label() {
        let result = ExtendedParser::parse("MATCH (u:User:Admin) REMOVE u:Admin");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Remove(stmt) => {
                assert_eq!(stmt.items.len(), 1);
                match &stmt.items[0] {
                    RemoveItem::Label { variable, label } => {
                        assert_eq!(variable.name, "u");
                        assert_eq!(label.name, "Admin");
                    }
                    _ => panic!("Expected label removal"),
                }
            }
            _ => panic!("Expected Remove statement"),
        }
    }

    #[test]
    fn parse_cypher_remove_multiple_items() {
        let result = ExtendedParser::parse(
            "MATCH (u:User) WHERE u.id = 1 REMOVE u.temp1, u.temp2, u:Temporary",
        );
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Remove(stmt) => {
                assert!(stmt.where_clause.is_some());
                assert_eq!(stmt.items.len(), 3);
            }
            _ => panic!("Expected Remove statement"),
        }
    }

    #[test]
    fn parse_cypher_remove_with_return() {
        let result = ExtendedParser::parse("MATCH (u:User) REMOVE u.temp RETURN u");
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let stmts = result.unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Statement::Remove(stmt) => {
                assert_eq!(stmt.return_clause.len(), 1);
            }
            _ => panic!("Expected Remove statement"),
        }
    }

    // ========== List Comprehension Parser Tests ==========

    #[test]
    fn parse_list_literal_empty() {
        let result = ExtendedParser::parse_list_or_comprehension("[]").unwrap();
        match result {
            Expr::ListLiteral(items) => {
                assert!(items.is_empty());
            }
            _ => panic!("Expected ListLiteral"),
        }
    }

    #[test]
    fn parse_list_literal_simple() {
        let result = ExtendedParser::parse_list_or_comprehension("[1, 2, 3]").unwrap();
        match result {
            Expr::ListLiteral(items) => {
                assert_eq!(items.len(), 3);
            }
            _ => panic!("Expected ListLiteral"),
        }
    }

    #[test]
    fn parse_list_literal_strings() {
        let result = ExtendedParser::parse_list_or_comprehension("['a', 'b', 'c']").unwrap();
        match result {
            Expr::ListLiteral(items) => {
                assert_eq!(items.len(), 3);
            }
            _ => panic!("Expected ListLiteral"),
        }
    }

    #[test]
    fn parse_list_comprehension_filter_only() {
        // [x IN list WHERE x > 2]
        let result =
            ExtendedParser::parse_list_or_comprehension("[x IN numbers WHERE x > 2]").unwrap();
        match result {
            Expr::ListComprehension { variable, filter_predicate, transform_expr, .. } => {
                assert_eq!(variable.name, "x");
                assert!(filter_predicate.is_some());
                assert!(transform_expr.is_none());
            }
            _ => panic!("Expected ListComprehension, got {:?}", result),
        }
    }

    #[test]
    fn parse_list_comprehension_transform_only() {
        // [x IN list | x * 2]
        let result = ExtendedParser::parse_list_or_comprehension("[x IN numbers | x * 2]").unwrap();
        match result {
            Expr::ListComprehension { variable, filter_predicate, transform_expr, .. } => {
                assert_eq!(variable.name, "x");
                assert!(filter_predicate.is_none());
                assert!(transform_expr.is_some());
            }
            _ => panic!("Expected ListComprehension, got {:?}", result),
        }
    }

    #[test]
    fn parse_list_comprehension_filter_and_transform() {
        // [x IN list WHERE x > 2 | x * x]
        let result =
            ExtendedParser::parse_list_or_comprehension("[x IN numbers WHERE x > 2 | x * x]")
                .unwrap();
        match result {
            Expr::ListComprehension { variable, filter_predicate, transform_expr, .. } => {
                assert_eq!(variable.name, "x");
                assert!(filter_predicate.is_some());
                assert!(transform_expr.is_some());
            }
            _ => panic!("Expected ListComprehension, got {:?}", result),
        }
    }

    #[test]
    fn parse_list_comprehension_no_filter_no_transform() {
        // [x IN list] - just iteration
        let result = ExtendedParser::parse_list_or_comprehension("[x IN numbers]").unwrap();
        match result {
            Expr::ListComprehension { variable, filter_predicate, transform_expr, .. } => {
                assert_eq!(variable.name, "x");
                assert!(filter_predicate.is_none());
                assert!(transform_expr.is_none());
            }
            _ => panic!("Expected ListComprehension, got {:?}", result),
        }
    }

    #[test]
    fn parse_list_comprehension_with_function_call() {
        // [x IN range(1, 10) WHERE x > 5]
        let result =
            ExtendedParser::parse_list_or_comprehension("[x IN range(1, 10) WHERE x > 5]").unwrap();
        match result {
            Expr::ListComprehension { variable, filter_predicate, .. } => {
                assert_eq!(variable.name, "x");
                assert!(filter_predicate.is_some());
            }
            _ => panic!("Expected ListComprehension, got {:?}", result),
        }
    }

    #[test]
    fn parse_nested_list_literal() {
        // [[1, 2], [3, 4]]
        let result = ExtendedParser::parse_list_or_comprehension("[[1, 2], [3, 4]]").unwrap();
        match result {
            Expr::ListLiteral(items) => {
                assert_eq!(items.len(), 2);
                match &items[0] {
                    Expr::ListLiteral(inner) => assert_eq!(inner.len(), 2),
                    _ => panic!("Expected nested ListLiteral"),
                }
            }
            _ => panic!("Expected ListLiteral"),
        }
    }
}

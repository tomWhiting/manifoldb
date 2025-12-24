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
    BinaryOp, CreateCollectionStatement, DistanceMetric, EdgeDirection, EdgeLength, EdgePattern,
    Expr, GraphPattern, Identifier, MatchStatement, NodePattern, OrderByExpr, ParameterRef,
    PathPattern, PropertyCondition, QualifiedName, ReturnItem, SelectStatement,
    ShortestPathPattern, Statement, VectorDef, VectorTypeDef, WeightSpec,
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

        // Step 1: Extract MATCH clauses (they're not valid SQL)
        let (sql_without_match, match_patterns) = Self::extract_match_clauses(input)?;

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
        upper.contains("RETURN")
    }

    /// Checks if the input is a CREATE COLLECTION statement.
    fn is_create_collection(input: &str) -> bool {
        let upper = input.trim().to_uppercase();
        upper.starts_with("CREATE COLLECTION")
            || upper.starts_with("CREATE IF NOT EXISTS COLLECTION")
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

        let vector_defs_str = &after_name[1..close_paren];

        // Parse vector definitions
        let vectors = Self::parse_vector_definitions(vector_defs_str)?;

        if vectors.is_empty() {
            return Err(ParseError::SqlSyntax(
                "CREATE COLLECTION requires at least one vector definition".to_string(),
            ));
        }

        let stmt = CreateCollectionStatement { if_not_exists, name, vectors };

        Ok(vec![Statement::CreateCollection(Box::new(stmt))])
    }

    /// Parses the vector definitions inside CREATE COLLECTION parentheses.
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
            if !limit_str.is_empty() {
                Some(Self::parse_limit_expr(limit_str)?)
            } else {
                None
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

    /// Extracts MATCH clauses from the SQL and returns the modified SQL.
    fn extract_match_clauses(input: &str) -> ParseResult<(String, Vec<GraphPattern>)> {
        let mut result = String::with_capacity(input.len());
        let mut patterns = Vec::new();
        let mut remaining = input;

        while let Some(match_pos) = Self::find_match_keyword(remaining) {
            // Add everything before MATCH
            result.push_str(&remaining[..match_pos]);

            // Find the end of the MATCH clause (WHERE, ORDER BY, GROUP BY, LIMIT, or ;)
            let after_match = &remaining[match_pos + 5..]; // Skip "MATCH"
            let end_pos = Self::find_match_end(after_match);

            let pattern_str = after_match[..end_pos].trim();
            let pattern = Self::parse_graph_pattern(pattern_str)?;
            patterns.push(pattern);

            remaining = &after_match[end_pos..];
        }

        result.push_str(remaining);

        Ok((result, patterns))
    }

    /// Finds the position of the MATCH keyword (case-insensitive, word boundary).
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
                return Some(absolute_pos);
            }

            search_from = absolute_pos + 5;
        }

        None
    }

    /// Finds the end of a MATCH clause.
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
        ];

        let mut min_pos = input.len();

        for keyword in &keywords {
            if let Some(pos) = input_upper.find(keyword) {
                // Check word boundary
                let before_ok = pos == 0 || !input.as_bytes()[pos - 1].is_ascii_alphanumeric();
                if before_ok && pos < min_pos {
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
        let op_pos = Self::find_operator(&chars, &op_chars);
        if op_pos.is_none() {
            return input.to_string();
        }
        let op_pos = op_pos.unwrap();

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
                    // Handle array subscripts - skip to matching '['
                    pos -= 1;
                    while pos > 0 && chars[pos - 1] != '[' {
                        pos -= 1;
                    }
                    pos = pos.saturating_sub(1);
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
                    // Handle array subscripts
                    pos += 1;
                    while pos < chars.len() && chars[pos] != ']' {
                        pos += 1;
                    }
                    if pos < chars.len() {
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
    /// Transforms `__VEC_EUCLIDEAN__(a, b)` to `Expr::BinaryOp { left: a, op: EuclideanDistance, right: b }`.
    fn convert_vector_function(expr: &mut Expr) {
        let replacement = if let Expr::Function(func) = expr {
            let func_name = func.name.name().map(|id| id.name.as_str()).unwrap_or("");

            // Check for HYBRID function first
            if func_name.eq_ignore_ascii_case("HYBRID") || func_name.eq_ignore_ascii_case("RRF") {
                Self::parse_hybrid_function(func, func_name.eq_ignore_ascii_case("RRF"))
            } else {
                // Check for vector distance operators
                let op = match func_name {
                    "__VEC_EUCLIDEAN__" => Some(BinaryOp::EuclideanDistance),
                    "__VEC_COSINE__" => Some(BinaryOp::CosineDistance),
                    "__VEC_INNER__" => Some(BinaryOp::InnerProduct),
                    "__VEC_MAXSIM__" => Some(BinaryOp::MaxSim),
                    _ => None,
                };

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
        let (sql, patterns) = ExtendedParser::extract_match_clauses(
            "SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1",
        )
        .unwrap();

        assert!(sql.contains("SELECT * FROM users"));
        assert!(sql.contains("WHERE u.id = 1"));
        assert!(!sql.to_uppercase().contains("MATCH"));
        assert_eq!(patterns.len(), 1);
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
}

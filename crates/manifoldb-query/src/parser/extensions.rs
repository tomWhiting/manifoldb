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
    BinaryOp, DistanceMetric, EdgeDirection, EdgeLength, EdgePattern, Expr, GraphPattern,
    Identifier, NodePattern, ParameterRef, PathPattern, PropertyCondition, QualifiedName,
    SelectStatement, ShortestPathPattern, Statement, WeightSpec,
};
use crate::error::{ParseError, ParseResult};
use crate::parser::sql;

/// Extended SQL parser with graph and vector support.
pub struct ExtendedParser;

impl ExtendedParser {
    /// Parses an extended SQL query with graph and vector syntax.
    ///
    /// This function handles:
    /// 1. Pre-processing to convert custom operators to function calls
    /// 2. Standard SQL parsing
    /// 3. Post-processing to extract MATCH clauses
    /// 4. Restoration of vector operators from function calls
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is syntactically invalid.
    pub fn parse(input: &str) -> ParseResult<Vec<Statement>> {
        if input.trim().is_empty() {
            return Err(ParseError::EmptyQuery);
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
        } else {
            None
        };

        if let Some(new_expr) = replacement {
            *expr = new_expr;
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
}

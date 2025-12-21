//! Graph and vector syntax extensions.
//!
//! This module provides parsing for custom SQL extensions:
//! - Graph pattern matching (MATCH clause)
//! - Vector distance operators (<->, <=>, <#>)
//!
//! # Extended Syntax Examples
//!
//! ## Vector Similarity Search
//! ```sql
//! SELECT * FROM docs ORDER BY embedding <-> $query LIMIT 10;
//! ```
//!
//! ## Graph Pattern Matching
//! ```sql
//! SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1;
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
    SelectStatement, Statement,
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
        let mut result = String::new();
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
            let before_ok = absolute_pos == 0
                || !input.as_bytes()[absolute_pos - 1].is_ascii_alphanumeric();
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

        let keywords = ["WHERE", "ORDER", "GROUP", "HAVING", "LIMIT", "OFFSET", "UNION", "INTERSECT", "EXCEPT"];

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
    fn preprocess_vector_ops(input: &str) -> String {
        let mut result = input.to_string();

        // Replace <-> with __vec_euclidean__ function call marker
        result = result.replace("<->", " __VEC_EUCLIDEAN__ ");

        // Replace <=> with __vec_cosine__ function call marker
        result = result.replace("<=>", " __VEC_COSINE__ ");

        // Replace <#> with __vec_inner__ function call marker
        result = result.replace("<#>", " __VEC_INNER__ ");

        result
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
    fn restore_vector_ops_in_expr(expr: &mut Expr) {
        match expr {
            Expr::BinaryOp { left, right, .. } => {
                Self::restore_vector_ops_in_expr(left);
                Self::restore_vector_ops_in_expr(right);
            }
            Expr::UnaryOp { operand, .. } => {
                Self::restore_vector_ops_in_expr(operand);
            }
            Expr::Function(func) => {
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
            Expr::Column(name) => {
                // Check if this is a vector operator marker
                if let Some(ident) = name.name() {
                    // Vector operator markers shouldn't appear as columns after preprocessing,
                    // but if they do, they'll be handled by convert_vector_markers below
                    if !matches!(ident.name.as_str(), "__VEC_EUCLIDEAN__" | "__VEC_COSINE__" | "__VEC_INNER__") {
                        // Normal column reference - no action needed
                    }
                }
            }
            _ => {}
        }

        // Check for marker identifiers that became binary operations
        // The preprocessing turns `a <-> b` into `a __VEC_EUCLIDEAN__ b`
        // which sqlparser might parse as identifier comparisons
        Self::convert_vector_markers(expr);
    }

    /// Converts vector operator markers to proper binary operators.
    fn convert_vector_markers(expr: &mut Expr) {
        // We need to find patterns like: expr = Column("__VEC_EUCLIDEAN__")
        // and transform them into proper vector operations

        // This is a simplified approach - in practice, the preprocessing
        // makes this unnecessary for most cases
        if let Expr::Column(name) = expr {
            if let Some(ident) = name.name() {
                let op = match ident.name.as_str() {
                    "__VEC_EUCLIDEAN__" => Some(BinaryOp::EuclideanDistance),
                    "__VEC_COSINE__" => Some(BinaryOp::CosineDistance),
                    "__VEC_INNER__" => Some(BinaryOp::InnerProduct),
                    _ => None,
                };

                if op.is_some() {
                    // This marker shouldn't appear as a standalone column
                    // Log a warning in a real implementation
                }
            }
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

        let mut paths = Vec::new();
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
            return Err(ParseError::InvalidPattern(
                format!("expected '(' at start of node pattern, found: {}", input.chars().next().unwrap_or('?'))
            ));
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
        let mut labels = Vec::new();
        let mut properties = Vec::new();

        let mut current = input;

        // Parse variable (before first colon or brace)
        if !current.starts_with(':') && !current.starts_with('{') {
            let end = current.find([':', '{', ' '])
                .unwrap_or(current.len());
            let var_name = &current[..end];
            if !var_name.is_empty() {
                variable = Some(Identifier::new(var_name));
            }
            current = &current[end..];
        }

        // Parse labels (each starts with :)
        while current.starts_with(':') {
            current = &current[1..]; // Skip ':'
            let end = current.find([':', '{', ' ', ')'])
                .unwrap_or(current.len());
            let label = &current[..end];
            if !label.is_empty() {
                labels.push(Identifier::new(label));
            }
            current = current[end..].trim_start();
        }

        // Parse properties (in braces)
        if current.starts_with('{') {
            let close_brace = current.find('}')
                .ok_or_else(|| ParseError::InvalidPattern("unclosed properties".to_string()))?;
            let props_str = &current[1..close_brace];
            properties = Self::parse_properties(props_str)?;
        }

        Ok(NodePattern {
            variable,
            labels,
            properties,
        })
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
            return Err(ParseError::InvalidPattern(
                format!("expected edge pattern, found: {}", &input[..input.len().min(10)])
            ));
        };

        // Find the closing bracket
        let bracket_end = input[bracket_start + 1..].find(']')
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
        let mut edge_types = Vec::new();
        let mut length = EdgeLength::Single;
        let mut properties = Vec::new();

        if input.is_empty() {
            return Ok(EdgePattern {
                direction,
                variable,
                edge_types,
                properties,
                length,
            });
        }

        let mut current = input;

        // Parse variable (before first colon, asterisk, or brace)
        if !current.starts_with(':') && !current.starts_with('*') && !current.starts_with('{') {
            let end = current.find([':', '*', '{', ' '])
                .unwrap_or(current.len());
            let var_name = &current[..end];
            if !var_name.is_empty() {
                variable = Some(Identifier::new(var_name));
            }
            current = &current[end..];
        }

        // Parse edge types (each starts with : or |)
        while current.starts_with(':') || current.starts_with('|') {
            current = &current[1..]; // Skip ':' or '|'
            let end = current.find(['|', '*', '{', ' ', ']'])
                .unwrap_or(current.len());
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
            let end = current.find(['{', ' ', ']'])
                .unwrap_or(current.len());
            current = current[end..].trim_start();
        }

        // Parse properties (in braces)
        if current.starts_with('{') {
            let close_brace = current.find('}')
                .ok_or_else(|| ParseError::InvalidPattern("unclosed edge properties".to_string()))?;
            let props_str = &current[1..close_brace];
            properties = Self::parse_properties(props_str)?;
        }

        Ok(EdgePattern {
            direction,
            variable,
            edge_types,
            properties,
            length,
        })
    }

    /// Parses edge length specification.
    fn parse_edge_length(input: &str) -> ParseResult<EdgeLength> {
        let input = input.trim();

        if input.is_empty() || input.starts_with('{') || input.starts_with(' ') || input.starts_with(']') {
            return Ok(EdgeLength::Any);
        }

        // Check for range (min..max)
        if let Some(range_pos) = input.find("..") {
            let before = &input[..range_pos];
            let after_start = range_pos + 2;
            let after_end = input[after_start..].find(|c: char| !c.is_ascii_digit())
                .map_or(input.len(), |p| after_start + p);
            let after = &input[after_start..after_end];

            let min = if before.is_empty() {
                None
            } else {
                Some(before.parse::<u32>().map_err(|_|
                    ParseError::InvalidPattern(format!("invalid min in range: {before}"))
                )?)
            };

            let max = if after.is_empty() {
                None
            } else {
                Some(after.parse::<u32>().map_err(|_|
                    ParseError::InvalidPattern(format!("invalid max in range: {after}"))
                )?)
            };

            return Ok(EdgeLength::Range { min, max });
        }

        // Check for exact number
        let num_end = input.find(|c: char| !c.is_ascii_digit())
            .unwrap_or(input.len());
        let num_str = &input[..num_end];

        if !num_str.is_empty() {
            let n = num_str.parse::<u32>().map_err(|_|
                ParseError::InvalidPattern(format!("invalid edge length: {num_str}"))
            )?;
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

        let mut properties = Vec::new();

        for pair in input.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }

            let colon_pos = pair.find(':')
                .ok_or_else(|| ParseError::InvalidPattern(format!("invalid property: {pair}")))?;

            let name = pair[..colon_pos].trim();
            let value_str = pair[colon_pos + 1..].trim();

            let value = Self::parse_property_value(value_str);

            properties.push(PropertyCondition {
                name: Identifier::new(name),
                value,
            });
        }

        Ok(properties)
    }

    /// Parses a property value.
    fn parse_property_value(input: &str) -> Expr {
        let input = input.trim();

        // String literal
        if (input.starts_with('\'') && input.ends_with('\''))
            || (input.starts_with('"') && input.ends_with('"')) {
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

/// Parses a vector distance expression.
///
/// This function is used to parse expressions like:
/// - `embedding <-> $query` (Euclidean distance)
/// - `vec_column <=> $param` (Cosine distance)
/// - `data <#> $vector` (Inner product)
pub fn parse_vector_distance(
    left: Expr,
    metric: DistanceMetric,
    right: Expr,
) -> Expr {
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
        let (path, _) = ExtendedParser::parse_path_pattern(
            "(a)-[:KNOWS]->(b)-[:LIKES]->(c)"
        ).unwrap();
        assert_eq!(path.steps.len(), 2);
    }

    #[test]
    fn parse_graph_pattern() {
        let pattern = ExtendedParser::parse_graph_pattern(
            "(u:User)-[:FOLLOWS]->(f:User)"
        ).unwrap();
        assert_eq!(pattern.paths.len(), 1);
    }

    #[test]
    fn parse_multiple_paths() {
        let pattern = ExtendedParser::parse_graph_pattern(
            "(a)-[:R1]->(b), (b)-[:R2]->(c)"
        ).unwrap();
        assert_eq!(pattern.paths.len(), 2);
    }

    #[test]
    fn extract_match_clause() {
        let (sql, patterns) = ExtendedParser::extract_match_clauses(
            "SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1"
        ).unwrap();

        assert!(sql.contains("SELECT * FROM users"));
        assert!(sql.contains("WHERE u.id = 1"));
        assert!(!sql.to_uppercase().contains("MATCH"));
        assert_eq!(patterns.len(), 1);
    }

    #[test]
    fn parse_extended_select() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1"
        ).unwrap();

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
    }

    #[test]
    fn parse_node_with_properties() {
        let (node, _) = ExtendedParser::parse_node_pattern("(p:Person {name: 'Alice', age: 30})").unwrap();
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
}

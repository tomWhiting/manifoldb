//! Comprehensive parser tests for `manifoldb-query`.
//!
//! These tests verify parsing of:
//! - Standard SQL statements
//! - Extended graph syntax (MATCH clauses)
//! - Vector operations (distance operators)
//! - Combined queries
//! - Error handling

use manifoldb_query::ast::{
    BinaryOp, DataType, EdgeDirection, EdgeLength, Expr, JoinType, Literal, ParameterRef,
    SelectItem, Statement, TableRef,
};
use manifoldb_query::error::ParseError;
use manifoldb_query::parser::{parse_single_statement, parse_sql, ExtendedParser};

// ============================================================================
// Standard SQL Parsing Tests
// ============================================================================

mod standard_sql {
    use super::*;

    #[test]
    fn parse_simple_select() {
        let stmt = parse_single_statement("SELECT * FROM users").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.projection.len(), 1);
                assert!(matches!(select.projection[0], SelectItem::Wildcard));
                assert_eq!(select.from.len(), 1);
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_select_columns() {
        let stmt = parse_single_statement("SELECT id, name, email FROM users").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.projection.len(), 3);
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_select_with_alias() {
        let stmt = parse_single_statement("SELECT u.name AS user_name FROM users u").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { alias, .. } = &select.projection[0] {
                    assert!(alias.is_some());
                    assert_eq!(alias.as_ref().unwrap().name, "user_name");
                } else {
                    panic!("expected aliased expression");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_select_with_where() {
        let stmt = parse_single_statement("SELECT * FROM users WHERE id = 1 AND active = true").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert!(select.where_clause.is_some());
                if let Some(Expr::BinaryOp { op: BinaryOp::And, .. }) = select.where_clause {
                    // Good - it's an AND expression
                } else {
                    panic!("expected AND expression in WHERE");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_select_with_order_by() {
        let stmt = parse_single_statement("SELECT * FROM users ORDER BY name ASC, created_at DESC").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.order_by.len(), 2);
                assert!(select.order_by[0].asc);
                assert!(!select.order_by[1].asc);
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_select_with_limit_offset() {
        let stmt = parse_single_statement("SELECT * FROM users LIMIT 10 OFFSET 20").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert!(select.limit.is_some());
                assert!(select.offset.is_some());
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_select_distinct() {
        let stmt = parse_single_statement("SELECT DISTINCT name FROM users").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert!(select.distinct);
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_inner_join() {
        let stmt = parse_single_statement(
            "SELECT u.name, o.total FROM users u INNER JOIN orders o ON u.id = o.user_id"
        ).unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.from.len(), 1);
                if let TableRef::Join(join) = &select.from[0] {
                    assert_eq!(join.join_type, JoinType::Inner);
                } else {
                    panic!("expected JOIN");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_left_join() {
        let stmt = parse_single_statement(
            "SELECT * FROM users u LEFT JOIN orders o ON u.id = o.user_id"
        ).unwrap();
        match stmt {
            Statement::Select(select) => {
                if let TableRef::Join(join) = &select.from[0] {
                    assert_eq!(join.join_type, JoinType::LeftOuter);
                } else {
                    panic!("expected JOIN");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_group_by_having() {
        let stmt = parse_single_statement(
            "SELECT category, COUNT(*) FROM products GROUP BY category HAVING COUNT(*) > 5"
        ).unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.group_by.len(), 1);
                assert!(select.having.is_some());
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_subquery() {
        let stmt = parse_single_statement(
            "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)"
        ).unwrap();
        match stmt {
            Statement::Select(select) => {
                if let Some(Expr::InSubquery { .. }) = select.where_clause {
                    // Good
                } else {
                    panic!("expected IN subquery");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_insert_values() {
        let stmt = parse_single_statement(
            "INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com')"
        ).unwrap();
        match stmt {
            Statement::Insert(insert) => {
                assert_eq!(insert.columns.len(), 2);
            }
            _ => panic!("expected INSERT"),
        }
    }

    #[test]
    fn parse_insert_multiple_rows() {
        let stmt = parse_single_statement(
            "INSERT INTO users (name) VALUES ('Alice'), ('Bob'), ('Charlie')"
        ).unwrap();
        match stmt {
            Statement::Insert(insert) => {
                if let manifoldb_query::ast::InsertSource::Values(rows) = &insert.source {
                    assert_eq!(rows.len(), 3);
                } else {
                    panic!("expected VALUES");
                }
            }
            _ => panic!("expected INSERT"),
        }
    }

    #[test]
    fn parse_update() {
        let stmt = parse_single_statement(
            "UPDATE users SET name = 'Bob', email = 'bob@example.com' WHERE id = 1"
        ).unwrap();
        match stmt {
            Statement::Update(update) => {
                assert_eq!(update.assignments.len(), 2);
                assert!(update.where_clause.is_some());
            }
            _ => panic!("expected UPDATE"),
        }
    }

    #[test]
    fn parse_delete() {
        let stmt = parse_single_statement("DELETE FROM users WHERE id = 1").unwrap();
        match stmt {
            Statement::Delete(delete) => {
                assert!(delete.where_clause.is_some());
            }
            _ => panic!("expected DELETE"),
        }
    }

    #[test]
    fn parse_create_table() {
        let stmt = parse_single_statement(
            "CREATE TABLE users (
                id BIGINT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP
            )"
        ).unwrap();
        match stmt {
            Statement::CreateTable(create) => {
                assert_eq!(create.columns.len(), 4);
                assert_eq!(create.columns[0].data_type, DataType::BigInt);
            }
            _ => panic!("expected CREATE TABLE"),
        }
    }

    #[test]
    fn parse_create_index() {
        let stmt = parse_single_statement(
            "CREATE INDEX idx_users_email ON users (email)"
        ).unwrap();
        match stmt {
            Statement::CreateIndex(create) => {
                assert_eq!(create.name.name, "idx_users_email");
                assert_eq!(create.columns.len(), 1);
            }
            _ => panic!("expected CREATE INDEX"),
        }
    }

    #[test]
    fn parse_drop_table() {
        let stmt = parse_single_statement("DROP TABLE IF EXISTS users CASCADE").unwrap();
        match stmt {
            Statement::DropTable(drop) => {
                assert!(drop.if_exists);
                assert!(drop.cascade);
            }
            _ => panic!("expected DROP TABLE"),
        }
    }

    #[test]
    fn parse_explain() {
        let stmt = parse_single_statement("EXPLAIN SELECT * FROM users").unwrap();
        match stmt {
            Statement::Explain(inner) => {
                assert!(matches!(*inner, Statement::Select(_)));
            }
            _ => panic!("expected EXPLAIN"),
        }
    }

    #[test]
    fn parse_multiple_statements() {
        let stmts = parse_sql("SELECT 1; SELECT 2; SELECT 3").unwrap();
        assert_eq!(stmts.len(), 3);
    }
}

// ============================================================================
// Expression Parsing Tests
// ============================================================================

mod expressions {
    use super::*;

    #[test]
    fn parse_literal_integer() {
        let stmt = parse_single_statement("SELECT 42").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr: Expr::Literal(Literal::Integer(42)), .. } = &select.projection[0] {
                    // Good
                } else {
                    panic!("expected integer literal 42");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_literal_float() {
        let stmt = parse_single_statement("SELECT 3.14").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr: Expr::Literal(Literal::Float(f)), .. } = &select.projection[0] {
                    assert!((f - 3.14).abs() < 0.001);
                } else {
                    panic!("expected float literal");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_literal_string() {
        let stmt = parse_single_statement("SELECT 'hello world'").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr: Expr::Literal(Literal::String(s)), .. } = &select.projection[0] {
                    assert_eq!(s, "hello world");
                } else {
                    panic!("expected string literal");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_boolean_true() {
        let stmt = parse_single_statement("SELECT true").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr: Expr::Literal(Literal::Boolean(true)), .. } = &select.projection[0] {
                    // Good
                } else {
                    panic!("expected boolean true");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_null() {
        let stmt = parse_single_statement("SELECT NULL").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr: Expr::Literal(Literal::Null), .. } = &select.projection[0] {
                    // Good
                } else {
                    panic!("expected NULL");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_arithmetic() {
        let stmt = parse_single_statement("SELECT 1 + 2 * 3").unwrap();
        match stmt {
            Statement::Select(select) => {
                // Due to operator precedence, this should be 1 + (2 * 3)
                if let SelectItem::Expr { expr: Expr::BinaryOp { op: BinaryOp::Add, .. }, .. } = &select.projection[0] {
                    // Good - top-level is Add
                } else {
                    panic!("expected binary operation");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_comparison() {
        let stmt = parse_single_statement("SELECT * FROM t WHERE a >= b").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let Some(Expr::BinaryOp { op: BinaryOp::GtEq, .. }) = &select.where_clause {
                    // Good
                } else {
                    panic!("expected >= comparison");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_between() {
        let stmt = parse_single_statement("SELECT * FROM t WHERE x BETWEEN 1 AND 10").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let Some(Expr::Between { negated: false, .. }) = &select.where_clause {
                    // Good
                } else {
                    panic!("expected BETWEEN");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_like() {
        let stmt = parse_single_statement("SELECT * FROM t WHERE name LIKE '%test%'").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let Some(Expr::BinaryOp { op: BinaryOp::Like, .. }) = &select.where_clause {
                    // Good
                } else {
                    panic!("expected LIKE");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_is_null() {
        let stmt = parse_single_statement("SELECT * FROM t WHERE x IS NULL").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let Some(Expr::UnaryOp { op: manifoldb_query::ast::UnaryOp::IsNull, .. }) = &select.where_clause {
                    // Good
                } else {
                    panic!("expected IS NULL");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_function_call() {
        let stmt = parse_single_statement("SELECT COUNT(*), SUM(amount) FROM orders").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.projection.len(), 2);
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_case_expression() {
        let stmt = parse_single_statement(
            "SELECT CASE WHEN x > 0 THEN 'positive' ELSE 'non-positive' END FROM t"
        ).unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr: Expr::Case(_), .. } = &select.projection[0] {
                    // Good
                } else {
                    panic!("expected CASE expression");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_cast() {
        let stmt = parse_single_statement("SELECT CAST(x AS INTEGER) FROM t").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr: Expr::Cast { .. }, .. } = &select.projection[0] {
                    // Good
                } else {
                    panic!("expected CAST");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_positional_parameter() {
        let stmt = parse_single_statement("SELECT * FROM users WHERE id = $1").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let Some(Expr::BinaryOp { right, .. }) = &select.where_clause {
                    if let Expr::Parameter(ParameterRef::Positional(1)) = right.as_ref() {
                        // Good
                    } else {
                        panic!("expected positional parameter $1");
                    }
                } else {
                    panic!("expected binary op");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_anonymous_parameter() {
        let stmt = parse_single_statement("SELECT * FROM users WHERE id = ?").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let Some(Expr::BinaryOp { right, .. }) = &select.where_clause {
                    if let Expr::Parameter(ParameterRef::Anonymous) = right.as_ref() {
                        // Good
                    } else {
                        panic!("expected anonymous parameter ?");
                    }
                } else {
                    panic!("expected binary op");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }
}

// ============================================================================
// Graph Pattern Parsing Tests
// ============================================================================

mod graph_patterns {
    use super::*;

    #[test]
    fn parse_simple_match() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1"
        ).unwrap();

        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            assert!(select.match_clause.is_some());
            let pattern = select.match_clause.as_ref().unwrap();
            assert_eq!(pattern.paths.len(), 1);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_with_labels() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM users MATCH (u:User)-[:FOLLOWS]->(f:User)"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            let path = &pattern.paths[0];
            assert_eq!(path.start.labels.len(), 1);
            assert_eq!(path.start.labels[0].name, "User");
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_multiple_labels() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM t MATCH (p:Person:Employee)"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            assert_eq!(pattern.paths[0].start.labels.len(), 2);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_with_variable() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM t MATCH (a)-[r:KNOWS]->(b)"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            let edge = &pattern.paths[0].steps[0].0;
            assert_eq!(edge.variable.as_ref().unwrap().name, "r");
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_undirected() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM t MATCH (a)-[:KNOWS]-(b)"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            let edge = &pattern.paths[0].steps[0].0;
            assert_eq!(edge.direction, EdgeDirection::Undirected);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_left_direction() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM t MATCH (a)<-[:CREATED_BY]-(b)"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            let edge = &pattern.paths[0].steps[0].0;
            assert_eq!(edge.direction, EdgeDirection::Left);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_variable_length() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM t MATCH (a)-[:PATH*1..5]->(b)"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            let edge = &pattern.paths[0].steps[0].0;
            assert_eq!(edge.length, EdgeLength::Range { min: Some(1), max: Some(5) });
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_any_length() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM t MATCH (a)-[:PATH*]->(b)"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            let edge = &pattern.paths[0].steps[0].0;
            assert_eq!(edge.length, EdgeLength::Any);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_min_only() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM t MATCH (a)-[:PATH*2..]->(b)"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            let edge = &pattern.paths[0].steps[0].0;
            assert_eq!(edge.length, EdgeLength::Range { min: Some(2), max: None });
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_long_path() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM t MATCH (a)-[:R1]->(b)-[:R2]->(c)-[:R3]->(d)"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            assert_eq!(pattern.paths[0].steps.len(), 3);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_multiple_paths() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM t MATCH (a)-[:R1]->(b), (b)-[:R2]->(c)"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            assert_eq!(pattern.paths.len(), 2);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_multiple_edge_types() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM t MATCH (a)-[:KNOWS|FOLLOWS]->(b)"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            let edge = &pattern.paths[0].steps[0].0;
            assert_eq!(edge.edge_types.len(), 2);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_with_properties() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM t MATCH (p:Person {name: 'Alice'})-[:KNOWS]->(f)"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            assert_eq!(pattern.paths[0].start.properties.len(), 1);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_with_where() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1 AND f.active = true"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            assert!(select.match_clause.is_some());
            assert!(select.where_clause.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_with_order_limit() {
        let stmts = ExtendedParser::parse(
            "SELECT * FROM t MATCH (a)-[:R]->(b) ORDER BY a.name LIMIT 10"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            assert!(select.match_clause.is_some());
            assert_eq!(select.order_by.len(), 1);
            assert!(select.limit.is_some());
        } else {
            panic!("expected SELECT");
        }
    }
}

// ============================================================================
// Vector Operations Tests
// ============================================================================

mod vector_ops {
    use super::*;

    // Note: Full vector operator parsing requires preprocessing.
    // These tests verify the AST structures are correct.

    #[test]
    fn binary_op_euclidean_exists() {
        // Verify the Euclidean distance operator is in the AST
        assert_eq!(BinaryOp::EuclideanDistance.to_string(), "<->");
    }

    #[test]
    fn binary_op_cosine_exists() {
        assert_eq!(BinaryOp::CosineDistance.to_string(), "<=>");
    }

    #[test]
    fn binary_op_inner_product_exists() {
        assert_eq!(BinaryOp::InnerProduct.to_string(), "<#>");
    }

    #[test]
    fn distance_metric_functions() {
        use manifoldb_query::ast::DistanceMetric;

        assert_eq!(DistanceMetric::Euclidean.function_name(), "euclidean_distance");
        assert_eq!(DistanceMetric::Cosine.function_name(), "cosine_distance");
        assert_eq!(DistanceMetric::InnerProduct.function_name(), "inner_product");
    }

    #[test]
    fn vector_search_params() {
        use manifoldb_query::ast::VectorSearchParams;

        let params = VectorSearchParams::new()
            .with_limit(10)
            .with_ef_search(100)
            .with_distance_threshold(0.5);

        assert_eq!(params.limit, Some(10));
        assert_eq!(params.ef_search, Some(100));
        assert_eq!(params.distance_threshold, Some(0.5));
    }
}

// ============================================================================
// Combined Query Tests
// ============================================================================

mod combined {
    use super::*;

    #[test]
    fn parse_graph_with_from() {
        let stmts = ExtendedParser::parse(
            "SELECT d.*, a.name FROM docs d MATCH (d)-[:AUTHORED_BY]->(a) WHERE d.id = 1"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            assert!(!select.from.is_empty());
            assert!(select.match_clause.is_some());
            assert!(select.where_clause.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_complex_query() {
        let stmts = ExtendedParser::parse(
            "SELECT u.name, COUNT(f.id) as follower_count
             FROM users u
             MATCH (u)-[:FOLLOWS]->(f)
             WHERE u.active = true
             GROUP BY u.name
             ORDER BY follower_count DESC
             LIMIT 10"
        ).unwrap();

        if let Statement::Select(select) = &stmts[0] {
            assert!(select.match_clause.is_some());
            assert!(select.where_clause.is_some());
            assert!(!select.group_by.is_empty());
            assert!(!select.order_by.is_empty());
            assert!(select.limit.is_some());
        } else {
            panic!("expected SELECT");
        }
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

mod errors {
    use super::*;

    #[test]
    fn empty_query_error() {
        let result = parse_sql("");
        assert!(matches!(result, Err(ParseError::EmptyQuery)));
    }

    #[test]
    fn whitespace_only_error() {
        let result = parse_sql("   \n\t  ");
        assert!(matches!(result, Err(ParseError::EmptyQuery)));
    }

    #[test]
    fn syntax_error() {
        let result = parse_sql("SELCT * FROM users");
        assert!(matches!(result, Err(ParseError::SqlSyntax(_))));
    }

    #[test]
    fn unclosed_parenthesis() {
        let result = parse_sql("SELECT * FROM users WHERE (id = 1");
        assert!(result.is_err());
    }

    #[test]
    fn missing_from() {
        // Some dialects allow this, so it might not be an error
        let result = parse_sql("SELECT *");
        // Just check it doesn't panic
        let _ = result;
    }

    #[test]
    fn invalid_pattern_unclosed() {
        let result = ExtendedParser::parse(
            "SELECT * FROM t MATCH (a"
        );
        assert!(result.is_err());
    }

    #[test]
    fn invalid_edge_pattern() {
        let result = ExtendedParser::parse(
            "SELECT * FROM t MATCH (a)-->(b)"
        );
        // This should fail - edges need brackets
        assert!(result.is_err());
    }
}

// ============================================================================
// AST Builder Tests
// ============================================================================

mod ast_builders {
    use manifoldb_query::ast::{
        Expr, Identifier, NodePattern, EdgePattern, PathPattern, QualifiedName,
        SelectItem, SelectStatement, TableRef,
    };

    #[test]
    fn build_select_programmatically() {
        let select = SelectStatement::new(vec![SelectItem::Wildcard])
            .from(TableRef::table(QualifiedName::simple("users")))
            .where_clause(
                Expr::column(QualifiedName::simple("id"))
                    .eq(Expr::integer(1))
            )
            .limit(Expr::integer(10));

        assert!(matches!(select.projection[0], SelectItem::Wildcard));
        assert!(select.where_clause.is_some());
        assert!(select.limit.is_some());
    }

    #[test]
    fn build_path_pattern() {
        let path = PathPattern::chain(
            NodePattern::with_label("a", "Person"),
            EdgePattern::directed().edge_type("KNOWS"),
            NodePattern::with_label("b", "Person"),
        ).then(
            EdgePattern::directed().edge_type("LIKES"),
            NodePattern::var("c"),
        );

        assert_eq!(path.steps.len(), 2);
        assert_eq!(path.start.labels[0].name, "Person");
    }

    #[test]
    fn build_expression_chain() {
        let expr = Expr::column(QualifiedName::simple("x"))
            .gt(Expr::integer(0))
            .and(
                Expr::column(QualifiedName::simple("y"))
                    .lt(Expr::integer(100))
            );

        match expr {
            Expr::BinaryOp { op: manifoldb_query::ast::BinaryOp::And, .. } => {}
            _ => panic!("expected AND"),
        }
    }

    #[test]
    fn qualified_name_parts() {
        let name = QualifiedName::qualified("schema", "table");
        assert_eq!(name.parts.len(), 2);
        assert_eq!(name.qualifiers().len(), 1);
        assert_eq!(name.name().unwrap().name, "table");
    }

    #[test]
    fn identifier_quoting() {
        let unquoted = Identifier::new("column");
        let quoted = Identifier::quoted("Column", '"');

        assert!(unquoted.quote_style.is_none());
        assert_eq!(quoted.quote_style, Some('"'));
        assert_eq!(quoted.to_string(), "\"Column\"");
    }
}

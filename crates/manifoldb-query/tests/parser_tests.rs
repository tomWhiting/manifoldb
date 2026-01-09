//! Comprehensive parser tests for `manifoldb-query`.
//!
//! These tests verify parsing of:
//! - Standard SQL statements
//! - Extended graph syntax (MATCH clauses)
//! - Vector operations (distance operators)
//! - Combined queries
//! - Error handling

use manifoldb_query::ast::{
    BinaryOp, DataType, EdgeDirection, EdgeLength, Expr, InsertSource, JoinType, Literal,
    ParameterRef, SelectItem, Statement, TableRef,
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
        let stmt =
            parse_single_statement("SELECT * FROM users WHERE id = 1 AND active = true").unwrap();
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
        let stmt = parse_single_statement("SELECT * FROM users ORDER BY name ASC, created_at DESC")
            .unwrap();
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
            "SELECT u.name, o.total FROM users u INNER JOIN orders o ON u.id = o.user_id",
        )
        .unwrap();
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
        let stmt =
            parse_single_statement("SELECT * FROM users u LEFT JOIN orders o ON u.id = o.user_id")
                .unwrap();
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
            "SELECT category, COUNT(*) FROM products GROUP BY category HAVING COUNT(*) > 5",
        )
        .unwrap();
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
        let stmt =
            parse_single_statement("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)")
                .unwrap();
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
            "INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com')",
        )
        .unwrap();
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
            "INSERT INTO users (name) VALUES ('Alice'), ('Bob'), ('Charlie')",
        )
        .unwrap();
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
            "UPDATE users SET name = 'Bob', email = 'bob@example.com' WHERE id = 1",
        )
        .unwrap();
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
            )",
        )
        .unwrap();
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
        let stmt = parse_single_statement("CREATE INDEX idx_users_email ON users (email)").unwrap();
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
                if let SelectItem::Expr { expr: Expr::Literal(Literal::Integer(42)), .. } =
                    &select.projection[0]
                {
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
        let stmt = parse_single_statement("SELECT 1.5").unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr: Expr::Literal(Literal::Float(f)), .. } =
                    &select.projection[0]
                {
                    assert!((f - 1.5).abs() < 0.001);
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
                if let SelectItem::Expr { expr: Expr::Literal(Literal::String(s)), .. } =
                    &select.projection[0]
                {
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
                if let SelectItem::Expr { expr: Expr::Literal(Literal::Boolean(true)), .. } =
                    &select.projection[0]
                {
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
                if let SelectItem::Expr { expr: Expr::Literal(Literal::Null), .. } =
                    &select.projection[0]
                {
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
                if let SelectItem::Expr { expr: Expr::BinaryOp { op: BinaryOp::Add, .. }, .. } =
                    &select.projection[0]
                {
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
                if let Some(Expr::UnaryOp { op: manifoldb_query::ast::UnaryOp::IsNull, .. }) =
                    &select.where_clause
                {
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
            "SELECT CASE WHEN x > 0 THEN 'positive' ELSE 'non-positive' END FROM t",
        )
        .unwrap();
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
        let stmts =
            ExtendedParser::parse("SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1")
                .unwrap();

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
        let stmts =
            ExtendedParser::parse("SELECT * FROM users MATCH (u:User)-[:FOLLOWS]->(f:User)")
                .unwrap();

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
        let stmts = ExtendedParser::parse("SELECT * FROM t MATCH (p:Person:Employee)").unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            assert_eq!(pattern.paths[0].start.labels.len(), 2);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_with_variable() {
        let stmts = ExtendedParser::parse("SELECT * FROM t MATCH (a)-[r:KNOWS]->(b)").unwrap();

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
        let stmts = ExtendedParser::parse("SELECT * FROM t MATCH (a)-[:KNOWS]-(b)").unwrap();

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
        let stmts = ExtendedParser::parse("SELECT * FROM t MATCH (a)<-[:CREATED_BY]-(b)").unwrap();

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
        let stmts = ExtendedParser::parse("SELECT * FROM t MATCH (a)-[:PATH*1..5]->(b)").unwrap();

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
        let stmts = ExtendedParser::parse("SELECT * FROM t MATCH (a)-[:PATH*]->(b)").unwrap();

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
        let stmts = ExtendedParser::parse("SELECT * FROM t MATCH (a)-[:PATH*2..]->(b)").unwrap();

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
        let stmts =
            ExtendedParser::parse("SELECT * FROM t MATCH (a)-[:R1]->(b)-[:R2]->(c)-[:R3]->(d)")
                .unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            assert_eq!(pattern.paths[0].steps.len(), 3);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_multiple_paths() {
        let stmts =
            ExtendedParser::parse("SELECT * FROM t MATCH (a)-[:R1]->(b), (b)-[:R2]->(c)").unwrap();

        if let Statement::Select(select) = &stmts[0] {
            let pattern = select.match_clause.as_ref().unwrap();
            assert_eq!(pattern.paths.len(), 2);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_multiple_edge_types() {
        let stmts =
            ExtendedParser::parse("SELECT * FROM t MATCH (a)-[:KNOWS|FOLLOWS]->(b)").unwrap();

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
        let stmts =
            ExtendedParser::parse("SELECT * FROM t MATCH (p:Person {name: 'Alice'})-[:KNOWS]->(f)")
                .unwrap();

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
            "SELECT * FROM users MATCH (u)-[:FOLLOWS]->(f) WHERE u.id = 1 AND f.active = true",
        )
        .unwrap();

        if let Statement::Select(select) = &stmts[0] {
            assert!(select.match_clause.is_some());
            assert!(select.where_clause.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_match_with_order_limit() {
        let stmts =
            ExtendedParser::parse("SELECT * FROM t MATCH (a)-[:R]->(b) ORDER BY a.name LIMIT 10")
                .unwrap();

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
    fn binary_op_maxsim_exists() {
        // Verify the MaxSim operator for ColBERT-style multi-vectors is in the AST
        assert_eq!(BinaryOp::MaxSim.to_string(), "<##>");
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

    #[test]
    fn parse_maxsim_operator() {
        // Parse a query using the MaxSim operator for ColBERT-style multi-vector search
        let stmts = ExtendedParser::parse(
            "SELECT * FROM docs ORDER BY colbert <##> $query_tokens LIMIT 10",
        )
        .unwrap();

        assert_eq!(stmts.len(), 1);
        if let Statement::Select(select) = &stmts[0] {
            // Check ORDER BY has the MaxSim operator
            assert_eq!(select.order_by.len(), 1);
            if let Expr::BinaryOp { op: BinaryOp::MaxSim, .. } = select.order_by[0].expr.as_ref() {
                // Good - MaxSim operator was parsed correctly
            } else {
                panic!("expected MaxSim binary operator in ORDER BY");
            }
            // Check LIMIT
            assert!(select.limit.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_maxsim_with_where() {
        // Parse MaxSim with a WHERE clause
        let stmts = ExtendedParser::parse(
            "SELECT id, title FROM documents WHERE category = 'science' ORDER BY embeddings <##> $query DESC LIMIT 5",
        )
        .unwrap();

        if let Statement::Select(select) = &stmts[0] {
            assert!(select.where_clause.is_some());
            assert_eq!(select.order_by.len(), 1);
            assert!(!select.order_by[0].asc); // DESC
            if let Expr::BinaryOp { op: BinaryOp::MaxSim, .. } = select.order_by[0].expr.as_ref() {
                // Good
            } else {
                panic!("expected MaxSim operator");
            }
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_maxsim_with_column_names() {
        // Parse MaxSim with specific column names
        let stmts = ExtendedParser::parse(
            "SELECT doc_id, content, token_embeddings <##> $query AS score FROM corpus ORDER BY score DESC LIMIT 20",
        )
        .unwrap();

        if let Statement::Select(select) = &stmts[0] {
            // Check projection includes the MaxSim expression with alias
            assert_eq!(select.projection.len(), 3);
            if let SelectItem::Expr { expr, alias } = &select.projection[2] {
                if let Expr::BinaryOp { op: BinaryOp::MaxSim, .. } = expr {
                    assert!(alias.is_some());
                    assert_eq!(alias.as_ref().unwrap().name, "score");
                } else {
                    panic!("expected MaxSim in projection");
                }
            } else {
                panic!("expected expression with alias");
            }
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_maxsim_with_multi_vector_literal() {
        // Parse MaxSim with a multi-vector literal instead of a parameter
        let stmts = ExtendedParser::parse(
            "SELECT * FROM docs ORDER BY embeddings <##> [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] LIMIT 10",
        )
        .unwrap();

        if let Statement::Select(select) = &stmts[0] {
            assert_eq!(select.order_by.len(), 1);
            if let Expr::BinaryOp { op: BinaryOp::MaxSim, right, .. } =
                select.order_by[0].expr.as_ref()
            {
                // The right operand should be a MultiVector literal
                if let Expr::Literal(Literal::MultiVector(v)) = right.as_ref() {
                    assert_eq!(v.len(), 2);
                    assert_eq!(v[0].len(), 3);
                    assert_eq!(v[1].len(), 3);
                } else {
                    panic!("expected MultiVector literal on right side, got {:?}", right);
                }
            } else {
                panic!("expected MaxSim operator in ORDER BY");
            }
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn parse_insert_with_multi_vector_literal() {
        // Parse INSERT with a multi-vector literal
        let stmts = ExtendedParser::parse(
            "INSERT INTO docs (id, embeddings) VALUES (1, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])",
        )
        .unwrap();

        if let Statement::Insert(insert) = &stmts[0] {
            assert_eq!(insert.columns.len(), 2);
            if let InsertSource::Values(rows) = &insert.source {
                assert_eq!(rows.len(), 1);
                // Second column value should be MultiVector
                if let Expr::Literal(Literal::MultiVector(v)) = &rows[0][1] {
                    assert_eq!(v.len(), 3);
                    assert_eq!(v[0].len(), 2);
                } else {
                    panic!("expected MultiVector literal");
                }
            } else {
                panic!("expected VALUES");
            }
        } else {
            panic!("expected INSERT");
        }
    }

    #[test]
    fn parse_maxsim_does_not_conflict_with_inner_product() {
        // Ensure <##> doesn't conflict with <#>
        let stmts = ExtendedParser::parse(
            "SELECT a <#> b AS inner_prod, c <##> d AS maxsim_score FROM vectors",
        )
        .unwrap();

        if let Statement::Select(select) = &stmts[0] {
            assert_eq!(select.projection.len(), 2);

            // First should be InnerProduct
            if let SelectItem::Expr { expr, alias } = &select.projection[0] {
                if let Expr::BinaryOp { op: BinaryOp::InnerProduct, .. } = expr {
                    assert_eq!(alias.as_ref().unwrap().name, "inner_prod");
                } else {
                    panic!("expected InnerProduct");
                }
            }

            // Second should be MaxSim
            if let SelectItem::Expr { expr, alias } = &select.projection[1] {
                if let Expr::BinaryOp { op: BinaryOp::MaxSim, .. } = expr {
                    assert_eq!(alias.as_ref().unwrap().name, "maxsim_score");
                } else {
                    panic!("expected MaxSim");
                }
            }
        } else {
            panic!("expected SELECT");
        }
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
            "SELECT d.*, a.name FROM docs d MATCH (d)-[:AUTHORED_BY]->(a) WHERE d.id = 1",
        )
        .unwrap();

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
             LIMIT 10",
        )
        .unwrap();

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
        let result = ExtendedParser::parse("SELECT * FROM t MATCH (a");
        assert!(result.is_err());
    }

    #[test]
    fn invalid_edge_pattern() {
        let result = ExtendedParser::parse("SELECT * FROM t MATCH (a)-->(b)");
        // This should fail - edges need brackets
        assert!(result.is_err());
    }
}

// ============================================================================
// AST Builder Tests
// ============================================================================

mod ast_builders {
    use manifoldb_query::ast::{
        EdgePattern, Expr, Identifier, NodePattern, PathPattern, QualifiedName, SelectItem,
        SelectStatement, TableRef,
    };

    #[test]
    fn build_select_programmatically() {
        let select = SelectStatement::new(vec![SelectItem::Wildcard])
            .from(TableRef::table(QualifiedName::simple("users")))
            .where_clause(Expr::column(QualifiedName::simple("id")).eq(Expr::integer(1)))
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
        )
        .then(EdgePattern::directed().edge_type("LIKES"), NodePattern::var("c"));

        assert_eq!(path.steps.len(), 2);
        assert_eq!(path.start.labels[0].name, "Person");
    }

    #[test]
    fn build_expression_chain() {
        let expr = Expr::column(QualifiedName::simple("x"))
            .gt(Expr::integer(0))
            .and(Expr::column(QualifiedName::simple("y")).lt(Expr::integer(100)));

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

// ============================================================================
// Standalone MATCH Statement Tests (Cypher-style)
// ============================================================================

mod standalone_match {
    use super::*;
    use manifoldb_query::ast::{ReturnItem, Statement};

    #[test]
    fn parse_simple_match_return() {
        let stmts =
            ExtendedParser::parse("MATCH (a:User)-[:FOLLOWS]->(b:User) RETURN a, b").unwrap();

        assert_eq!(stmts.len(), 1);
        if let Statement::Match(match_stmt) = &stmts[0] {
            // Check pattern
            assert_eq!(match_stmt.pattern.paths.len(), 1);
            assert_eq!(match_stmt.pattern.paths[0].start.labels[0].name, "User");

            // Check return clause
            assert_eq!(match_stmt.return_clause.len(), 2);
        } else {
            panic!("expected MATCH statement");
        }
    }

    #[test]
    fn parse_match_with_where() {
        let stmts = ExtendedParser::parse(
            "MATCH (a:User)-[:FOLLOWS]->(b:User) WHERE a.name = 'Alice' RETURN b.name",
        )
        .unwrap();

        if let Statement::Match(match_stmt) = &stmts[0] {
            assert!(match_stmt.where_clause.is_some());
        } else {
            panic!("expected MATCH statement");
        }
    }

    #[test]
    fn parse_match_return_distinct() {
        let stmts = ExtendedParser::parse("MATCH (a)-[:FOLLOWS]->(b) RETURN DISTINCT b").unwrap();

        if let Statement::Match(match_stmt) = &stmts[0] {
            assert!(match_stmt.distinct);
        } else {
            panic!("expected MATCH statement");
        }
    }

    #[test]
    fn parse_match_with_order_by() {
        let stmts =
            ExtendedParser::parse("MATCH (a)-[:FOLLOWS]->(b) RETURN b.name ORDER BY b.name DESC")
                .unwrap();

        if let Statement::Match(match_stmt) = &stmts[0] {
            assert_eq!(match_stmt.order_by.len(), 1);
            assert!(!match_stmt.order_by[0].asc); // DESC
        } else {
            panic!("expected MATCH statement");
        }
    }

    #[test]
    fn parse_match_with_skip_limit() {
        let stmts =
            ExtendedParser::parse("MATCH (a)-[:FOLLOWS]->(b) RETURN b SKIP 10 LIMIT 20").unwrap();

        if let Statement::Match(match_stmt) = &stmts[0] {
            assert!(match_stmt.skip.is_some());
            assert!(match_stmt.limit.is_some());
        } else {
            panic!("expected MATCH statement");
        }
    }

    #[test]
    fn parse_match_return_wildcard() {
        let stmts = ExtendedParser::parse("MATCH (a)-[:KNOWS]->(b) RETURN *").unwrap();

        if let Statement::Match(match_stmt) = &stmts[0] {
            assert_eq!(match_stmt.return_clause.len(), 1);
            assert!(matches!(match_stmt.return_clause[0], ReturnItem::Wildcard));
        } else {
            panic!("expected MATCH statement");
        }
    }

    #[test]
    fn parse_match_return_with_alias() {
        let stmts = ExtendedParser::parse("MATCH (a)-[:FOLLOWS]->(b) RETURN b.name AS friend_name")
            .unwrap();

        if let Statement::Match(match_stmt) = &stmts[0] {
            if let ReturnItem::Expr { alias, .. } = &match_stmt.return_clause[0] {
                assert!(alias.is_some());
                assert_eq!(alias.as_ref().unwrap().name, "friend_name");
            } else {
                panic!("expected expression return item");
            }
        } else {
            panic!("expected MATCH statement");
        }
    }

    #[test]
    fn parse_match_variable_length() {
        let stmts =
            ExtendedParser::parse("MATCH (a:User)-[:FOLLOWS*1..3]->(b:User) RETURN b").unwrap();

        if let Statement::Match(match_stmt) = &stmts[0] {
            let edge = &match_stmt.pattern.paths[0].steps[0].0;
            assert_eq!(edge.length, EdgeLength::Range { min: Some(1), max: Some(3) });
        } else {
            panic!("expected MATCH statement");
        }
    }

    #[test]
    fn parse_match_complex_pattern() {
        let stmts = ExtendedParser::parse(
            "MATCH (a:User)-[:WORKS_ON]->(p:Project)<-[:WORKS_ON]-(b:User) WHERE a.name = 'Alice' AND a <> b RETURN b.name, p.name",
        )
        .unwrap();

        if let Statement::Match(match_stmt) = &stmts[0] {
            // Check pattern has multiple steps
            assert_eq!(match_stmt.pattern.paths[0].steps.len(), 2);

            // Check return clause has 2 items
            assert_eq!(match_stmt.return_clause.len(), 2);
        } else {
            panic!("expected MATCH statement");
        }
    }

    #[test]
    fn parse_match_converts_to_select() {
        let stmts = ExtendedParser::parse(
            "MATCH (a)-[:FOLLOWS]->(b) WHERE a.id = 1 RETURN b.name ORDER BY b.name LIMIT 10",
        )
        .unwrap();

        if let Statement::Match(match_stmt) = &stmts[0] {
            // Convert to SELECT
            let select = match_stmt.to_select();

            // Verify the SELECT has all the parts
            assert!(select.match_clause.is_some());
            assert!(select.where_clause.is_some());
            assert!(!select.order_by.is_empty());
            assert!(select.limit.is_some());
        } else {
            panic!("expected MATCH statement");
        }
    }

    #[test]
    fn parse_match_multiple_paths() {
        let stmts = ExtendedParser::parse("MATCH (a)-[:R1]->(b), (b)-[:R2]->(c) RETURN c").unwrap();

        if let Statement::Match(match_stmt) = &stmts[0] {
            assert_eq!(match_stmt.pattern.paths.len(), 2);
        } else {
            panic!("expected MATCH statement");
        }
    }

    #[test]
    fn parse_match_full_syntax() {
        // Test full Cypher-style syntax as described in the task
        let stmts = ExtendedParser::parse(
            r"MATCH (a:User)-[:FOLLOWS]->(b:User)
               WHERE a.name = 'Alice'
               RETURN b.name",
        )
        .unwrap();

        if let Statement::Match(match_stmt) = &stmts[0] {
            // Pattern check
            assert_eq!(match_stmt.pattern.paths.len(), 1);

            // WHERE check
            assert!(match_stmt.where_clause.is_some());

            // RETURN check
            assert_eq!(match_stmt.return_clause.len(), 1);
        } else {
            panic!("expected MATCH statement");
        }
    }
}

// ============================================================================
// Common Table Expressions (CTE) Tests
// ============================================================================

mod cte {
    use super::*;

    #[test]
    fn parse_simple_cte() {
        let stmt = parse_single_statement(
            "WITH active_users AS (SELECT * FROM users WHERE status = 'active')
             SELECT * FROM active_users WHERE age > 21",
        )
        .unwrap();

        match stmt {
            Statement::Select(select) => {
                // Check WITH clause
                assert_eq!(select.with_clauses.len(), 1);
                assert_eq!(select.with_clauses[0].name.name, "active_users");
                assert!(select.with_clauses[0].columns.is_empty());

                // Check the CTE query
                let cte_query = &select.with_clauses[0].query;
                assert!(cte_query.where_clause.is_some());

                // Check the main query uses the CTE
                assert_eq!(select.from.len(), 1);
                if let TableRef::Table { name, .. } = &select.from[0] {
                    assert_eq!(name.name().unwrap().name, "active_users");
                } else {
                    panic!("expected table reference");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_multiple_ctes() {
        let stmt = parse_single_statement(
            "WITH
                dept_totals AS (SELECT dept_id, SUM(salary) as total FROM employees GROUP BY dept_id),
                high_spenders AS (SELECT * FROM dept_totals WHERE total > 100000)
             SELECT * FROM high_spenders",
        )
        .unwrap();

        match stmt {
            Statement::Select(select) => {
                // Check we have two CTEs
                assert_eq!(select.with_clauses.len(), 2);
                assert_eq!(select.with_clauses[0].name.name, "dept_totals");
                assert_eq!(select.with_clauses[1].name.name, "high_spenders");
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_cte_with_column_aliases() {
        let stmt =
            parse_single_statement("WITH temp(a, b, c) AS (SELECT 1, 2, 3) SELECT * FROM temp")
                .unwrap();

        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.with_clauses.len(), 1);
                assert_eq!(select.with_clauses[0].name.name, "temp");
                assert_eq!(select.with_clauses[0].columns.len(), 3);
                assert_eq!(select.with_clauses[0].columns[0].name, "a");
                assert_eq!(select.with_clauses[0].columns[1].name, "b");
                assert_eq!(select.with_clauses[0].columns[2].name, "c");
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_cte_with_join() {
        let stmt = parse_single_statement(
            "WITH user_orders AS (
                SELECT u.id, u.name, o.total
                FROM users u
                JOIN orders o ON u.id = o.user_id
             )
             SELECT * FROM user_orders WHERE total > 100",
        )
        .unwrap();

        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.with_clauses.len(), 1);

                // Check the CTE query has a join
                let cte_query = &select.with_clauses[0].query;
                assert_eq!(cte_query.from.len(), 1);
                if let TableRef::Join(_) = &cte_query.from[0] {
                    // Good - it's a join
                } else {
                    panic!("expected JOIN in CTE");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_cte_with_aggregation() {
        let stmt = parse_single_statement(
            "WITH summary AS (
                SELECT category, COUNT(*) as cnt, AVG(price) as avg_price
                FROM products
                GROUP BY category
             )
             SELECT * FROM summary ORDER BY cnt DESC",
        )
        .unwrap();

        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.with_clauses.len(), 1);
                assert_eq!(select.with_clauses[0].name.name, "summary");

                // Check the CTE query has GROUP BY
                let cte_query = &select.with_clauses[0].query;
                assert_eq!(cte_query.group_by.len(), 1);

                // Check the main query has ORDER BY
                assert_eq!(select.order_by.len(), 1);
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_cte_referenced_multiple_times() {
        let stmt = parse_single_statement(
            "WITH temp AS (SELECT id, value FROM data)
             SELECT t1.id, t1.value, t2.value
             FROM temp t1
             JOIN temp t2 ON t1.id = t2.id + 1",
        )
        .unwrap();

        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.with_clauses.len(), 1);
                assert_eq!(select.with_clauses[0].name.name, "temp");

                // Main query should have a join with two references to 'temp'
                assert_eq!(select.from.len(), 1);
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_cte_with_subquery() {
        let stmt = parse_single_statement(
            "WITH filtered AS (
                SELECT * FROM users WHERE id IN (SELECT user_id FROM premium_users)
             )
             SELECT * FROM filtered",
        )
        .unwrap();

        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.with_clauses.len(), 1);

                // Check the CTE query has a subquery in WHERE
                let cte_query = &select.with_clauses[0].query;
                assert!(cte_query.where_clause.is_some());
                if let Some(Expr::InSubquery { .. }) = &cte_query.where_clause {
                    // Good - it's an IN subquery
                } else {
                    panic!("expected IN subquery in CTE WHERE clause");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn recursive_cte_parses_successfully() {
        // WITH RECURSIVE is now supported
        let result = parse_single_statement(
            "WITH RECURSIVE cte AS (SELECT 1 as n UNION ALL SELECT n + 1 FROM cte WHERE n < 10)
             SELECT * FROM cte",
        );

        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.with_clauses.len(), 1);
                let cte = &select.with_clauses[0];
                assert_eq!(cte.name.name, "cte");
                assert!(cte.recursive, "CTE should be marked as recursive");
            }
            _ => panic!("expected SELECT statement"),
        }
    }

    #[test]
    fn recursive_cte_with_columns() {
        // WITH RECURSIVE with explicit column list
        let result = parse_single_statement(
            "WITH RECURSIVE hierarchy(id, parent_id, level) AS (
                SELECT id, parent_id, 1 FROM nodes WHERE parent_id IS NULL
                UNION ALL
                SELECT n.id, n.parent_id, h.level + 1 FROM nodes n JOIN hierarchy h ON n.parent_id = h.id
             )
             SELECT * FROM hierarchy",
        );

        assert!(result.is_ok());
        let stmt = result.unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.with_clauses.len(), 1);
                let cte = &select.with_clauses[0];
                assert_eq!(cte.name.name, "hierarchy");
                assert!(cte.recursive);
                assert_eq!(cte.columns.len(), 3);
                assert_eq!(cte.columns[0].name, "id");
                assert_eq!(cte.columns[1].name, "parent_id");
                assert_eq!(cte.columns[2].name, "level");
            }
            _ => panic!("expected SELECT statement"),
        }
    }
}

// ============================================================================
// Window Function Parsing Tests
// ============================================================================

// ============================================================================
// CALL/YIELD Statement Parsing Tests
// ============================================================================

mod call_yield {
    use super::*;
    use manifoldb_query::ast::{Statement, YieldItem};

    #[test]
    fn parse_simple_call() {
        let stmts = ExtendedParser::parse("CALL algo.pageRank()").unwrap();

        assert_eq!(stmts.len(), 1);
        if let Statement::Call(call) = &stmts[0] {
            assert_eq!(call.procedure_name.parts.len(), 2);
            assert_eq!(call.procedure_name.parts[0].name, "algo");
            assert_eq!(call.procedure_name.parts[1].name, "pageRank");
            assert!(call.arguments.is_empty());
            assert!(call.yield_items.is_empty());
            assert!(call.where_clause.is_none());
        } else {
            panic!("expected CALL statement");
        }
    }

    #[test]
    fn parse_call_with_arguments() {
        let stmts = ExtendedParser::parse("CALL algo.pageRank(0.85, 100)").unwrap();

        if let Statement::Call(call) = &stmts[0] {
            assert_eq!(call.arguments.len(), 2);
            // First arg should be a float literal
            if let Expr::Literal(Literal::Float(f)) = &call.arguments[0] {
                assert!((*f - 0.85).abs() < 0.001);
            } else {
                panic!("expected float literal for first argument");
            }
            // Second arg should be an integer literal
            if let Expr::Literal(Literal::Integer(i)) = &call.arguments[1] {
                assert_eq!(*i, 100);
            } else {
                panic!("expected integer literal for second argument");
            }
        } else {
            panic!("expected CALL statement");
        }
    }

    #[test]
    fn parse_call_with_yield() {
        let stmts = ExtendedParser::parse("CALL algo.pageRank() YIELD nodeId, score").unwrap();

        if let Statement::Call(call) = &stmts[0] {
            assert_eq!(call.yield_items.len(), 2);
            if let YieldItem::Column { name, alias } = &call.yield_items[0] {
                assert_eq!(name.name, "nodeId");
                assert!(alias.is_none());
            } else {
                panic!("expected column yield item");
            }
            if let YieldItem::Column { name, alias } = &call.yield_items[1] {
                assert_eq!(name.name, "score");
                assert!(alias.is_none());
            } else {
                panic!("expected column yield item");
            }
        } else {
            panic!("expected CALL statement");
        }
    }

    #[test]
    fn parse_call_with_yield_wildcard() {
        let stmts = ExtendedParser::parse("CALL algo.pageRank() YIELD *").unwrap();

        if let Statement::Call(call) = &stmts[0] {
            assert_eq!(call.yield_items.len(), 1);
            assert!(matches!(call.yield_items[0], YieldItem::Wildcard));
        } else {
            panic!("expected CALL statement");
        }
    }

    #[test]
    fn parse_call_with_yield_alias() {
        let stmts = ExtendedParser::parse("CALL algo.pageRank() YIELD nodeId AS id, score AS rank")
            .unwrap();

        if let Statement::Call(call) = &stmts[0] {
            assert_eq!(call.yield_items.len(), 2);
            if let YieldItem::Column { name, alias } = &call.yield_items[0] {
                assert_eq!(name.name, "nodeId");
                assert!(alias.is_some());
                assert_eq!(alias.as_ref().unwrap().name, "id");
            } else {
                panic!("expected column yield item");
            }
            if let YieldItem::Column { name, alias } = &call.yield_items[1] {
                assert_eq!(name.name, "score");
                assert!(alias.is_some());
                assert_eq!(alias.as_ref().unwrap().name, "rank");
            } else {
                panic!("expected column yield item");
            }
        } else {
            panic!("expected CALL statement");
        }
    }

    #[test]
    fn parse_call_with_yield_and_where() {
        let stmts =
            ExtendedParser::parse("CALL algo.pageRank() YIELD nodeId, score WHERE score > 0.1")
                .unwrap();

        if let Statement::Call(call) = &stmts[0] {
            assert_eq!(call.yield_items.len(), 2);
            assert!(call.where_clause.is_some());
            // Check the WHERE clause is a comparison
            if let Some(Expr::BinaryOp { op: BinaryOp::Gt, .. }) = &call.where_clause {
                // Good
            } else {
                panic!("expected > comparison in WHERE");
            }
        } else {
            panic!("expected CALL statement");
        }
    }

    #[test]
    fn parse_call_shortest_path() {
        let stmts =
            ExtendedParser::parse("CALL algo.shortestPath(1, 10) YIELD path, length").unwrap();

        if let Statement::Call(call) = &stmts[0] {
            assert_eq!(call.procedure_name.parts[0].name, "algo");
            assert_eq!(call.procedure_name.parts[1].name, "shortestPath");
            assert_eq!(call.arguments.len(), 2);
            assert_eq!(call.yield_items.len(), 2);
        } else {
            panic!("expected CALL statement");
        }
    }

    #[test]
    fn parse_call_with_string_argument() {
        let stmts =
            ExtendedParser::parse("CALL algo.shortestPath(1, 10, 'FRIEND') YIELD path").unwrap();

        if let Statement::Call(call) = &stmts[0] {
            assert_eq!(call.arguments.len(), 3);
            // Third arg should be a string literal
            if let Expr::Literal(Literal::String(s)) = &call.arguments[2] {
                assert_eq!(s, "FRIEND");
            } else {
                panic!("expected string literal for third argument");
            }
        } else {
            panic!("expected CALL statement");
        }
    }

    #[test]
    fn parse_call_with_parameter() {
        let stmts =
            ExtendedParser::parse("CALL algo.pageRank($damping) YIELD nodeId, score").unwrap();

        if let Statement::Call(call) = &stmts[0] {
            assert_eq!(call.arguments.len(), 1);
            // Arg should be a named parameter
            if let Expr::Parameter(ParameterRef::Named(name)) = &call.arguments[0] {
                assert_eq!(name, "damping");
            } else {
                panic!("expected named parameter");
            }
        } else {
            panic!("expected CALL statement");
        }
    }

    #[test]
    fn parse_call_simple_name() {
        let stmts = ExtendedParser::parse("CALL dbms.info() YIELD version").unwrap();

        if let Statement::Call(call) = &stmts[0] {
            assert_eq!(call.procedure_name.parts.len(), 2);
            assert_eq!(call.procedure_name.parts[0].name, "dbms");
            assert_eq!(call.procedure_name.parts[1].name, "info");
        } else {
            panic!("expected CALL statement");
        }
    }

    #[test]
    fn parse_call_no_yield() {
        // CALL without YIELD should still parse
        let stmts = ExtendedParser::parse("CALL db.index.rebuild('users')").unwrap();

        if let Statement::Call(call) = &stmts[0] {
            assert_eq!(call.arguments.len(), 1);
            assert!(call.yield_items.is_empty());
        } else {
            panic!("expected CALL statement");
        }
    }

    #[test]
    fn parse_call_nested_procedure_name() {
        let stmts = ExtendedParser::parse("CALL gds.graph.project.cypher('myGraph')").unwrap();

        if let Statement::Call(call) = &stmts[0] {
            // Should have 4 parts: gds.graph.project.cypher
            assert_eq!(call.procedure_name.parts.len(), 4);
            assert_eq!(call.procedure_name.parts[0].name, "gds");
            assert_eq!(call.procedure_name.parts[1].name, "graph");
            assert_eq!(call.procedure_name.parts[2].name, "project");
            assert_eq!(call.procedure_name.parts[3].name, "cypher");
        } else {
            panic!("expected CALL statement");
        }
    }
}

mod window_functions {
    use super::*;

    #[test]
    fn parse_row_number() {
        let stmt = parse_single_statement(
            "SELECT name, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn FROM employees",
        )
        .unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.projection.len(), 2);
                // The second item should be a function with OVER clause
                if let SelectItem::Expr { expr, alias } = &select.projection[1] {
                    assert!(alias.is_some());
                    assert_eq!(alias.as_ref().unwrap().name, "rn");
                    if let Expr::Function(func) = expr {
                        assert!(func.over.is_some());
                        let over = func.over.as_ref().unwrap();
                        assert!(over.partition_by.is_empty());
                        assert_eq!(over.order_by.len(), 1);
                        assert!(!over.order_by[0].asc); // DESC
                    } else {
                        panic!("expected Function expression");
                    }
                } else {
                    panic!("expected Expr projection");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_row_number_with_partition() {
        let stmt = parse_single_statement(
            "SELECT name, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) FROM employees",
        )
        .unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr, .. } = &select.projection[1] {
                    if let Expr::Function(func) = expr {
                        let over = func.over.as_ref().unwrap();
                        assert_eq!(over.partition_by.len(), 1); // PARTITION BY dept
                        assert_eq!(over.order_by.len(), 1); // ORDER BY salary DESC
                    } else {
                        panic!("expected Function expression");
                    }
                } else {
                    panic!("expected Expr projection");
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_rank() {
        let stmt = parse_single_statement(
            "SELECT name, RANK() OVER (ORDER BY score DESC) AS rank FROM scores",
        )
        .unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr, alias } = &select.projection[1] {
                    assert_eq!(alias.as_ref().unwrap().name, "rank");
                    if let Expr::Function(func) = expr {
                        assert!(func.over.is_some());
                        let func_name = func.name.parts.last().map(|p| p.name.to_uppercase());
                        assert_eq!(func_name, Some("RANK".to_string()));
                    } else {
                        panic!("expected Function expression");
                    }
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_dense_rank() {
        let stmt = parse_single_statement(
            "SELECT name, DENSE_RANK() OVER (ORDER BY score DESC) AS drank FROM scores",
        )
        .unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr, alias } = &select.projection[1] {
                    assert_eq!(alias.as_ref().unwrap().name, "drank");
                    if let Expr::Function(func) = expr {
                        assert!(func.over.is_some());
                        let func_name = func.name.parts.last().map(|p| p.name.to_uppercase());
                        assert_eq!(func_name, Some("DENSE_RANK".to_string()));
                    } else {
                        panic!("expected Function expression");
                    }
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_multiple_partition_columns() {
        let stmt = parse_single_statement(
            "SELECT ROW_NUMBER() OVER (PARTITION BY dept, region ORDER BY salary DESC) FROM employees",
        )
        .unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr, .. } = &select.projection[0] {
                    if let Expr::Function(func) = expr {
                        let over = func.over.as_ref().unwrap();
                        assert_eq!(over.partition_by.len(), 2); // dept, region
                    } else {
                        panic!("expected Function expression");
                    }
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_multiple_order_columns() {
        let stmt = parse_single_statement(
            "SELECT ROW_NUMBER() OVER (ORDER BY dept ASC, salary DESC) FROM employees",
        )
        .unwrap();
        match stmt {
            Statement::Select(select) => {
                if let SelectItem::Expr { expr, .. } = &select.projection[0] {
                    if let Expr::Function(func) = expr {
                        let over = func.over.as_ref().unwrap();
                        assert_eq!(over.order_by.len(), 2);
                        assert!(over.order_by[0].asc); // ASC
                        assert!(!over.order_by[1].asc); // DESC
                    } else {
                        panic!("expected Function expression");
                    }
                }
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_window_with_where() {
        let stmt = parse_single_statement(
            "SELECT name, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn
             FROM employees
             WHERE dept = 'Sales'",
        )
        .unwrap();
        match stmt {
            Statement::Select(select) => {
                assert!(select.where_clause.is_some());
                assert_eq!(select.projection.len(), 2);
            }
            _ => panic!("expected SELECT"),
        }
    }

    #[test]
    fn parse_multiple_window_functions() {
        let stmt = parse_single_statement(
            "SELECT name,
                    ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn,
                    RANK() OVER (ORDER BY salary DESC) AS rnk,
                    DENSE_RANK() OVER (ORDER BY salary DESC) AS drnk
             FROM employees",
        )
        .unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.projection.len(), 4); // name + 3 window functions

                // Check that all window functions have OVER clauses
                for i in 1..4 {
                    if let SelectItem::Expr { expr, .. } = &select.projection[i] {
                        if let Expr::Function(func) = expr {
                            assert!(
                                func.over.is_some(),
                                "window function {} should have OVER clause",
                                i
                            );
                        } else {
                            panic!("expected Function expression at position {}", i);
                        }
                    }
                }
            }
            _ => panic!("expected SELECT"),
        }
    }
}

// ============================================================================
// Procedure Infrastructure Tests
// ============================================================================

mod procedure_infrastructure {
    use std::sync::Arc;

    use manifoldb_core::Value;
    use manifoldb_query::exec::{Row, RowBatch, Schema};
    use manifoldb_query::procedure::{
        Procedure, ProcedureArgs, ProcedureError, ProcedureParameter, ProcedureRegistry,
        ProcedureResult, ProcedureSignature, ReturnColumn,
    };

    // Test procedure implementation that echoes its input
    struct EchoProcedure;

    impl Procedure for EchoProcedure {
        fn signature(&self) -> ProcedureSignature {
            ProcedureSignature::new("test.echo")
                .with_description("Echoes the input value")
                .with_parameter(ProcedureParameter::required("value", "STRING"))
                .with_return(ReturnColumn::new("result", "STRING"))
        }

        fn execute(&self, args: ProcedureArgs) -> ProcedureResult<RowBatch> {
            let schema = Arc::new(Schema::new(vec!["result".to_string()]));
            let mut batch = RowBatch::new(Arc::clone(&schema));

            if let Some(value) = args.get(0) {
                batch.push(Row::new(schema, vec![value.clone()]));
            }

            Ok(batch)
        }
    }

    #[test]
    fn registry_register_and_get() {
        let mut registry = ProcedureRegistry::new();
        registry.register(Arc::new(EchoProcedure));

        assert!(registry.contains("test.echo"));
        assert!(!registry.contains("test.unknown"));

        let proc = registry.get("test.echo");
        assert!(proc.is_some());
    }

    #[test]
    fn registry_get_or_error() {
        let mut registry = ProcedureRegistry::new();
        registry.register(Arc::new(EchoProcedure));

        assert!(registry.get_or_error("test.echo").is_ok());
        assert!(matches!(registry.get_or_error("unknown"), Err(ProcedureError::NotFound(_))));
    }

    #[test]
    fn registry_execute_procedure() {
        let mut registry = ProcedureRegistry::new();
        registry.register(Arc::new(EchoProcedure));

        let proc = registry.get("test.echo").unwrap();
        let args = ProcedureArgs::new(vec![Value::from("hello")]);
        let result = proc.execute(args).expect("execution should succeed");

        assert_eq!(result.len(), 1);
        assert_eq!(result.rows()[0].get(0), Some(&Value::from("hello")));
    }

    #[test]
    fn registry_list_procedures() {
        let mut registry = ProcedureRegistry::new();
        registry.register(Arc::new(EchoProcedure));

        let names = registry.list_names();
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"test.echo"));

        let sigs = registry.list_signatures();
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].name, "test.echo");
    }

    #[test]
    fn registry_unregister() {
        let mut registry = ProcedureRegistry::new();
        registry.register(Arc::new(EchoProcedure));

        assert!(registry.contains("test.echo"));
        registry.unregister("test.echo");
        assert!(!registry.contains("test.echo"));
    }

    #[test]
    fn procedure_args_get_string() {
        let args = ProcedureArgs::new(vec![Value::from("test"), Value::from(42i64)]);

        assert_eq!(args.get_string(0, "param").ok(), Some("test"));
        assert!(args.get_string(1, "param").is_err()); // Wrong type
        assert!(args.get_string(2, "param").is_err()); // Out of bounds
    }

    #[test]
    fn procedure_args_get_int() {
        let args = ProcedureArgs::new(vec![Value::from(42i64)]);

        assert_eq!(args.get_int(0, "param").ok(), Some(42));
        assert!(args.get_int(1, "param").is_err()); // Out of bounds
    }

    #[test]
    fn procedure_args_get_float() {
        let args = ProcedureArgs::new(vec![Value::from(3.14f64), Value::from(42i64)]);

        assert_eq!(args.get_float(0, "param").ok(), Some(3.14));
        // Integer promotion to float
        assert_eq!(args.get_float(1, "param").ok(), Some(42.0));
    }

    #[test]
    fn procedure_args_get_float_or_default() {
        let args = ProcedureArgs::new(vec![Value::from(0.5f64)]);

        assert_eq!(args.get_float_or(0, 0.85), 0.5);
        assert_eq!(args.get_float_or(1, 0.85), 0.85); // Default
    }

    #[test]
    fn signature_parameter_count() {
        let sig = ProcedureSignature::new("test.proc")
            .with_parameter(ProcedureParameter::required("a", "STRING"))
            .with_parameter(ProcedureParameter::required("b", "INTEGER"))
            .with_parameter(ProcedureParameter::optional("c", "FLOAT"));

        assert_eq!(sig.required_param_count(), 2);
        assert_eq!(sig.parameters.len(), 3);
    }

    #[test]
    fn signature_validate_arg_count_success() {
        let sig = ProcedureSignature::new("test.proc")
            .with_parameter(ProcedureParameter::required("name", "STRING"));

        assert!(sig.validate_arg_count(1).is_ok());
    }

    #[test]
    fn signature_validate_arg_count_missing_required() {
        let sig = ProcedureSignature::new("test.proc")
            .with_parameter(ProcedureParameter::required("name", "STRING"))
            .with_parameter(ProcedureParameter::required("value", "INTEGER"));

        // Missing second required arg
        assert!(sig.validate_arg_count(1).is_err());
    }

    #[test]
    fn return_column_description() {
        let col = ReturnColumn::new("score", "FLOAT").with_description("The computed score");

        assert_eq!(col.name, "score");
        assert_eq!(col.type_hint, "FLOAT");
        assert_eq!(col.description, "The computed score");
    }
}

// ============================================================================
// Built-in Procedure Tests
// ============================================================================

mod builtin_procedures {
    use manifoldb_query::procedure::{
        register_builtins, PageRankProcedure, Procedure, ProcedureRegistry, ShortestPathProcedure,
    };

    #[test]
    fn pagerank_signature() {
        let proc = PageRankProcedure;
        let sig = proc.signature();

        assert_eq!(sig.name, "algo.pageRank");
        assert_eq!(sig.parameters.len(), 2);
        assert_eq!(sig.returns.len(), 2);
        assert!(!sig.description.is_empty());
    }

    #[test]
    fn pagerank_output_schema() {
        let proc = PageRankProcedure;
        let schema = proc.output_schema();

        assert_eq!(schema.columns(), vec!["nodeId", "score"]);
    }

    #[test]
    fn pagerank_requires_context() {
        let proc = PageRankProcedure;
        assert!(proc.requires_context());
    }

    #[test]
    fn shortest_path_signature() {
        let proc = ShortestPathProcedure;
        let sig = proc.signature();

        assert_eq!(sig.name, "algo.shortestPath");
        assert_eq!(sig.parameters.len(), 4);
        assert_eq!(sig.required_param_count(), 2);
        assert_eq!(sig.returns.len(), 2);
    }

    #[test]
    fn shortest_path_output_schema() {
        let proc = ShortestPathProcedure;
        let schema = proc.output_schema();

        assert_eq!(schema.columns(), vec!["path", "length"]);
    }

    #[test]
    fn shortest_path_requires_context() {
        let proc = ShortestPathProcedure;
        assert!(proc.requires_context());
    }

    #[test]
    fn register_builtins_adds_procedures() {
        let mut registry = ProcedureRegistry::new();
        register_builtins(&mut registry);

        // Centrality algorithms
        assert!(registry.contains("algo.pageRank"));
        assert!(registry.contains("algo.betweennessCentrality"));
        assert!(registry.contains("algo.closenessCentrality"));
        assert!(registry.contains("algo.degreeCentrality"));
        assert!(registry.contains("algo.eigenvectorCentrality"));

        // Community detection
        assert!(registry.contains("algo.labelPropagation"));
        assert!(registry.contains("algo.connectedComponents"));
        assert!(registry.contains("algo.stronglyConnectedComponents"));

        // Path finding
        assert!(registry.contains("algo.shortestPath"));

        assert_eq!(registry.len(), 9);
    }
}

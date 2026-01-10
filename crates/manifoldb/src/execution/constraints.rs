//! Constraint enforcement for DML operations.
//!
//! This module provides runtime validation of table constraints during
//! INSERT, UPDATE, and DELETE operations.
//!
//! # Supported Constraints
//!
//! - **CHECK**: Validate that expressions evaluate to true
//! - **FOREIGN KEY**: Validate referential integrity
//!
//! # Usage
//!
//! ```ignore
//! use manifoldb::execution::constraints::ConstraintValidator;
//!
//! // Validate constraints before INSERT
//! ConstraintValidator::validate_insert(&tx, "users", &values)?;
//! ```

use std::collections::HashMap;

use manifoldb_core::{Entity, Value};
use manifoldb_query::plan::logical::LogicalExpr;
use manifoldb_query::ExecutionContext;
use manifoldb_storage::Transaction;

use crate::schema::{SchemaManager, StoredColumnConstraint, StoredTableConstraint, TableSchema};
use crate::transaction::DatabaseTransaction;

/// Errors that can occur during constraint validation.
#[derive(Debug, thiserror::Error)]
pub enum ConstraintError {
    /// CHECK constraint violation.
    #[error("CHECK constraint violation: {0}")]
    CheckViolation(String),

    /// FOREIGN KEY reference not found.
    #[error("FOREIGN KEY violation: key ({values}) in table '{source_table}' not found in referenced table '{ref_table}'")]
    ForeignKeyNotFound {
        /// Values that were not found
        values: String,
        /// Source table
        source_table: String,
        /// Referenced table
        ref_table: String,
    },

    /// FOREIGN KEY would be orphaned by DELETE.
    #[error("FOREIGN KEY violation: cannot delete from '{table}', row is referenced by table '{referencing_table}'")]
    ForeignKeyOrphaned {
        /// Table being deleted from
        table: String,
        /// Table that references this row
        referencing_table: String,
    },

    /// Schema lookup error.
    #[error("Schema error: {0}")]
    Schema(String),

    /// Expression evaluation error.
    #[error("Expression error: {0}")]
    Expression(String),
}

/// Result type for constraint operations.
pub type ConstraintResult<T> = Result<T, ConstraintError>;

/// Constraint validator for DML operations.
pub struct ConstraintValidator;

impl ConstraintValidator {
    /// Validate all constraints before an INSERT operation.
    ///
    /// Checks:
    /// - CHECK constraints on columns and table
    /// - FOREIGN KEY references exist in referenced tables
    ///
    /// If the table has no schema (no constraints defined), validation succeeds.
    pub fn validate_insert<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        table: &str,
        column_values: &HashMap<String, Value>,
        ctx: &ExecutionContext,
    ) -> ConstraintResult<()> {
        // If table has no schema, there are no constraints to validate
        let schema = match Self::try_get_schema(tx, table)? {
            Some(s) => s,
            None => return Ok(()),
        };

        // Validate CHECK constraints
        Self::validate_check_constraints(tx, &schema, column_values, ctx)?;

        // Validate FOREIGN KEY references
        Self::validate_foreign_key_references(tx, &schema, column_values)?;

        Ok(())
    }

    /// Validate all constraints before an UPDATE operation.
    ///
    /// Similar to INSERT, checks CHECK and FOREIGN KEY constraints
    /// on the new values.
    ///
    /// If the table has no schema (no constraints defined), validation succeeds.
    pub fn validate_update<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        table: &str,
        old_entity: &Entity,
        new_values: &HashMap<String, Value>,
        ctx: &ExecutionContext,
    ) -> ConstraintResult<()> {
        // If table has no schema, there are no constraints to validate
        let schema = match Self::try_get_schema(tx, table)? {
            Some(s) => s,
            None => return Ok(()),
        };

        // Merge old values with new values (new values override)
        let mut merged = old_entity.properties.clone();
        for (k, v) in new_values {
            merged.insert(k.clone(), v.clone());
        }

        // Validate CHECK constraints on merged values
        Self::validate_check_constraints(tx, &schema, &merged, ctx)?;

        // Validate FOREIGN KEY references on the columns being updated
        Self::validate_foreign_key_references(tx, &schema, new_values)?;

        Ok(())
    }

    /// Validate FOREIGN KEY constraints before a DELETE operation.
    ///
    /// Checks that no other tables reference the row being deleted.
    /// Currently implements RESTRICT behavior (fails if referenced).
    pub fn validate_delete<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        table: &str,
        entity: &Entity,
    ) -> ConstraintResult<()> {
        // Find all tables that might reference this table
        let all_tables =
            SchemaManager::list_tables(tx).map_err(|e| ConstraintError::Schema(e.to_string()))?;

        for other_table in all_tables {
            if other_table == table {
                continue;
            }

            let other_schema = match SchemaManager::get_table(tx, &other_table)
                .map_err(|e| ConstraintError::Schema(e.to_string()))?
            {
                Some(s) => s,
                None => continue,
            };

            // Check if any table constraint references our table
            for constraint in &other_schema.constraints {
                if let StoredTableConstraint::ForeignKey {
                    columns, ref_table, ref_columns, ..
                } = constraint
                {
                    if ref_table == table {
                        // Check if any row in other_table references this entity
                        if Self::has_referencing_rows(
                            tx,
                            &other_table,
                            columns,
                            entity,
                            ref_columns,
                        )? {
                            return Err(ConstraintError::ForeignKeyOrphaned {
                                table: table.to_string(),
                                referencing_table: other_table,
                            });
                        }
                    }
                }
            }

            // Check column-level foreign keys
            for col in &other_schema.columns {
                for constraint in &col.constraints {
                    if let StoredColumnConstraint::ForeignKey {
                        table: ref_table_name,
                        column: ref_column,
                    } = constraint
                    {
                        if ref_table_name == table {
                            // Check if any row in other_table references this entity
                            let ref_col = if ref_column.is_empty() {
                                // If no column specified, assume same column name
                                col.name.clone()
                            } else {
                                ref_column.clone()
                            };

                            if Self::has_referencing_rows(
                                tx,
                                &other_table,
                                std::slice::from_ref(&col.name),
                                entity,
                                &[ref_col],
                            )? {
                                return Err(ConstraintError::ForeignKeyOrphaned {
                                    table: table.to_string(),
                                    referencing_table: other_table,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Try to get table schema, returning None if the table has no schema.
    fn try_get_schema<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        table: &str,
    ) -> ConstraintResult<Option<TableSchema>> {
        SchemaManager::get_table(tx, table).map_err(|e| ConstraintError::Schema(e.to_string()))
    }

    /// Validate CHECK constraints on column values.
    fn validate_check_constraints<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        schema: &TableSchema,
        values: &HashMap<String, Value>,
        ctx: &ExecutionContext,
    ) -> ConstraintResult<()> {
        // Check column-level CHECK constraints
        for col in &schema.columns {
            for constraint in &col.constraints {
                if let StoredColumnConstraint::Check { expression } = constraint {
                    Self::validate_check_expression(
                        tx,
                        expression,
                        values,
                        ctx,
                        &format!("column '{}'", col.name),
                    )?;
                }
            }
        }

        // Check table-level CHECK constraints
        for constraint in &schema.constraints {
            if let StoredTableConstraint::Check { expression, name } = constraint {
                let constraint_desc =
                    name.as_ref().map_or_else(|| "unnamed".to_string(), |n| format!("'{}'", n));
                Self::validate_check_expression(
                    tx,
                    expression,
                    values,
                    ctx,
                    &format!("constraint {}", constraint_desc),
                )?;
            }
        }

        Ok(())
    }

    /// Validate a single CHECK expression.
    fn validate_check_expression<T: Transaction>(
        _tx: &DatabaseTransaction<T>,
        expression_sql: &str,
        values: &HashMap<String, Value>,
        ctx: &ExecutionContext,
        constraint_desc: &str,
    ) -> ConstraintResult<()> {
        // Parse the expression from SQL
        let expr = Self::parse_check_expression(expression_sql)?;

        // Create a temporary entity with the values for evaluation
        let mut temp_entity = Entity::new(manifoldb_core::EntityId::new(0));
        for (k, v) in values {
            temp_entity.set_property(k, v.clone());
        }

        // Evaluate the expression
        let result = super::executor::evaluate_predicate_for_constraint(&expr, &temp_entity, ctx);

        if !result {
            return Err(ConstraintError::CheckViolation(format!(
                "{} failed: {}",
                constraint_desc, expression_sql
            )));
        }

        Ok(())
    }

    /// Parse a CHECK expression from its SQL representation.
    fn parse_check_expression(expression_sql: &str) -> ConstraintResult<LogicalExpr> {
        // Use the parser to convert SQL expression to LogicalExpr
        manifoldb_query::parser::parse_check_expression(expression_sql)
            .map_err(|e| ConstraintError::Expression(format!("Failed to parse expression: {}", e)))
    }

    /// Validate FOREIGN KEY references exist.
    fn validate_foreign_key_references<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        schema: &TableSchema,
        values: &HashMap<String, Value>,
    ) -> ConstraintResult<()> {
        // Check column-level foreign keys
        for col in &schema.columns {
            for constraint in &col.constraints {
                if let StoredColumnConstraint::ForeignKey { table, column } = constraint {
                    if let Some(value) = values.get(&col.name) {
                        // Skip NULL values (NULLs are allowed in foreign keys)
                        if matches!(value, Value::Null) {
                            continue;
                        }

                        let ref_column = if column.is_empty() {
                            // If no column specified, assume same column name
                            col.name.clone()
                        } else {
                            column.clone()
                        };

                        if !Self::reference_exists(tx, table, &ref_column, value)? {
                            return Err(ConstraintError::ForeignKeyNotFound {
                                values: format!("{}={:?}", col.name, value),
                                source_table: schema.name.clone(),
                                ref_table: table.clone(),
                            });
                        }
                    }
                }
            }
        }

        // Check table-level foreign keys
        for constraint in &schema.constraints {
            if let StoredTableConstraint::ForeignKey { columns, ref_table, ref_columns, .. } =
                constraint
            {
                // Collect values for all FK columns
                let mut fk_values: Vec<(&str, &Value)> = Vec::new();
                let mut all_null = true;

                for (i, col) in columns.iter().enumerate() {
                    if let Some(value) = values.get(col) {
                        if !matches!(value, Value::Null) {
                            all_null = false;
                        }
                        let ref_col = ref_columns.get(i).map_or(col.as_str(), String::as_str);
                        fk_values.push((ref_col, value));
                    }
                }

                // Skip if all values are NULL (allowed by SQL standard)
                if all_null || fk_values.is_empty() {
                    continue;
                }

                if !Self::composite_reference_exists(tx, ref_table, &fk_values)? {
                    let values_str = fk_values
                        .iter()
                        .map(|(c, v)| format!("{}={:?}", c, v))
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(ConstraintError::ForeignKeyNotFound {
                        values: values_str,
                        source_table: schema.name.clone(),
                        ref_table: ref_table.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Check if a single column reference exists in the referenced table.
    fn reference_exists<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        ref_table: &str,
        ref_column: &str,
        value: &Value,
    ) -> ConstraintResult<bool> {
        // Iterate through entities to find a match
        let entities = tx
            .iter_entities(Some(ref_table))
            .map_err(|e| ConstraintError::Schema(format!("Failed to read table: {}", e)))?;

        for entity in entities {
            if let Some(ref_value) = entity.get_property(ref_column) {
                if Self::values_equal(value, ref_value) {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Check if a composite reference exists in the referenced table.
    fn composite_reference_exists<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        ref_table: &str,
        values: &[(&str, &Value)],
    ) -> ConstraintResult<bool> {
        let entities = tx
            .iter_entities(Some(ref_table))
            .map_err(|e| ConstraintError::Schema(format!("Failed to read table: {}", e)))?;

        'entity_loop: for entity in entities {
            for (col, expected_value) in values {
                // Skip NULL comparisons
                if matches!(expected_value, Value::Null) {
                    continue;
                }

                match entity.get_property(col) {
                    Some(actual_value) => {
                        if !Self::values_equal(expected_value, actual_value) {
                            continue 'entity_loop;
                        }
                    }
                    None => {
                        continue 'entity_loop;
                    }
                }
            }
            // All values matched
            return Ok(true);
        }

        Ok(false)
    }

    /// Check if any rows in a table reference the given entity.
    fn has_referencing_rows<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        referencing_table: &str,
        referencing_columns: &[String],
        entity: &Entity,
        referenced_columns: &[String],
    ) -> ConstraintResult<bool> {
        let entities = tx
            .iter_entities(Some(referencing_table))
            .map_err(|e| ConstraintError::Schema(format!("Failed to read table: {}", e)))?;

        'entity_loop: for referencing_entity in entities {
            for (i, ref_col) in referencing_columns.iter().enumerate() {
                let source_col = referenced_columns.get(i).map_or(ref_col.as_str(), String::as_str);

                let ref_value = match referencing_entity.get_property(ref_col) {
                    Some(v) => v,
                    None => continue 'entity_loop,
                };

                let source_value = match entity.get_property(source_col) {
                    Some(v) => v,
                    None => continue 'entity_loop,
                };

                // Skip NULL values
                if matches!(ref_value, Value::Null) || matches!(source_value, Value::Null) {
                    continue 'entity_loop;
                }

                if !Self::values_equal(ref_value, source_value) {
                    continue 'entity_loop;
                }
            }
            // All columns matched - found a referencing row
            return Ok(true);
        }

        Ok(false)
    }

    /// Compare two values for equality.
    fn values_equal(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Null, _) | (_, Value::Null) => false,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            // Cross-type numeric comparison
            (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => {
                (*a as f64 - b).abs() < f64::EPSILON
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_values_equal() {
        assert!(ConstraintValidator::values_equal(&Value::Int(42), &Value::Int(42)));
        assert!(!ConstraintValidator::values_equal(&Value::Int(42), &Value::Int(43)));
        assert!(ConstraintValidator::values_equal(
            &Value::String("test".into()),
            &Value::String("test".into())
        ));
        assert!(!ConstraintValidator::values_equal(&Value::Null, &Value::Int(42)));
    }
}

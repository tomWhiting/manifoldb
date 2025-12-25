//! Index scan execution for query processing.
//!
//! This module provides functions to execute index-based scans,
//! including point lookups and range scans using property indexes.

use manifoldb_core::index::{IndexId, PropertyIndexEntry, PropertyIndexScan};
use manifoldb_core::types::Value;
use manifoldb_core::Entity;
use manifoldb_query::exec::ExecutionContext;
use manifoldb_query::plan::{IndexRangeScanNode, IndexScanNode, LogicalExpr};
use manifoldb_storage::Transaction;

use crate::error::{Error, Result};
use crate::schema::SchemaManager;
use crate::transaction::DatabaseTransaction;

/// Evaluate a literal expression to a value.
///
/// This handles literals, parameters, and simple expressions that can be
/// evaluated without an entity context.
fn evaluate_literal_expr(expr: &LogicalExpr, ctx: &ExecutionContext) -> Result<Value> {
    use manifoldb_query::ast::Literal;

    match expr {
        LogicalExpr::Literal(lit) => match lit {
            Literal::Null => Ok(Value::Null),
            Literal::Boolean(b) => Ok(Value::Bool(*b)),
            Literal::Integer(i) => Ok(Value::Int(*i)),
            Literal::Float(f) => Ok(Value::Float(*f)),
            Literal::String(s) => Ok(Value::String(s.clone())),
            Literal::Vector(v) => Ok(Value::Vector(v.clone())),
            Literal::MultiVector(v) => Ok(Value::MultiVector(v.clone())),
        },
        LogicalExpr::Parameter(idx) => ctx
            .get_parameter(*idx as u32)
            .cloned()
            .ok_or_else(|| Error::Execution(format!("Parameter ${idx} not found"))),
        _ => Err(Error::Execution(format!("Cannot evaluate expression as literal: {expr:?}"))),
    }
}

/// Execute an index point lookup and return matching entities.
///
/// Uses the property index to find entities where the indexed column equals
/// a specific value, then fetches those entities by ID.
pub fn execute_index_scan<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    scan_node: &IndexScanNode,
    ctx: &ExecutionContext,
) -> Result<Vec<Entity>> {
    // Get the index schema to find the indexed column
    let index_schema = SchemaManager::get_index(tx, &scan_node.index_name)
        .map_err(|e| Error::Execution(format!("Failed to get index schema: {e}")))?
        .ok_or_else(|| Error::Execution(format!("Index '{}' not found", scan_node.index_name)))?;

    // Get the column name (first column for single-column index)
    let column = index_schema
        .columns
        .first()
        .ok_or_else(|| Error::Execution("Index has no columns".to_string()))?;

    let column_name = &column.expr;

    // Evaluate the key value
    let key_value = scan_node
        .key_values
        .first()
        .ok_or_else(|| Error::Execution("No key value specified".to_string()))?;

    let value = evaluate_literal_expr(key_value, ctx)?;

    // Create the index ID
    let index_id = IndexId::from_label_property(&scan_node.table_name, column_name);

    // Get the scan range for this exact value
    let (start, end) = PropertyIndexScan::exact_value_range(index_id, &value)
        .ok_or_else(|| Error::Execution("Value cannot be indexed".to_string()))?;

    // Scan the property index
    let keys = tx.scan_property_index(&start, &end).map_err(Error::Transaction)?;

    // Decode entity IDs from the index entries and fetch entities
    let mut entities = Vec::with_capacity(keys.len());
    for key in keys {
        if let Some(entry) = PropertyIndexEntry::decode_key(&key) {
            if let Some(entity) = tx.get_entity(entry.entity_id).map_err(Error::Transaction)? {
                entities.push(entity);
            }
        }
    }

    Ok(entities)
}

/// Execute an index range scan and return matching entities.
///
/// Uses the property index to find entities where the indexed column falls
/// within a range, then fetches those entities by ID.
pub fn execute_index_range_scan<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    scan_node: &IndexRangeScanNode,
    ctx: &ExecutionContext,
) -> Result<Vec<Entity>> {
    // Create the index ID
    let index_id = IndexId::from_label_property(&scan_node.table_name, &scan_node.key_column);

    // Evaluate bounds
    let lower_value =
        scan_node.lower_bound.as_ref().map(|expr| evaluate_literal_expr(expr, ctx)).transpose()?;

    let upper_value =
        scan_node.upper_bound.as_ref().map(|expr| evaluate_literal_expr(expr, ctx)).transpose()?;

    // Determine the scan range based on bounds
    // Note: PropertyIndexScan methods give us [start, end) ranges
    // We handle inclusivity at the filtering stage below
    let (start, end) = match (&lower_value, &upper_value) {
        (Some(low), Some(_high)) => {
            // For range between, we scan from lower to end of index
            // and filter by upper bound after fetching
            PropertyIndexScan::range_from(index_id, low)
                .ok_or_else(|| Error::Execution("Cannot create range scan".to_string()))?
        }
        (Some(low), None) => {
            // Range from lower bound to end
            PropertyIndexScan::range_from(index_id, low)
                .ok_or_else(|| Error::Execution("Cannot create range scan".to_string()))?
        }
        (None, Some(high)) => {
            // Range from start to upper bound (exclusive)
            PropertyIndexScan::range_to(index_id, high)
                .ok_or_else(|| Error::Execution("Cannot create range scan".to_string()))?
        }
        (None, None) => {
            // Full index scan
            PropertyIndexScan::full_index_range(index_id)
        }
    };

    // Scan the property index
    let keys = tx.scan_property_index(&start, &end).map_err(Error::Transaction)?;

    // Decode entity IDs and fetch entities
    // Apply inclusivity filtering
    let mut entities = Vec::with_capacity(keys.len());
    for key in keys {
        if let Some(entry) = PropertyIndexEntry::decode_key(&key) {
            // Check lower bound inclusivity
            if !scan_node.lower_inclusive {
                if let Some(ref lower) = lower_value {
                    if entry.value == *lower {
                        continue; // Skip the exact lower bound value
                    }
                }
            }

            // Check upper bound (range_from scans to end, so we need to filter)
            if let Some(ref upper) = upper_value {
                let cmp = compare_values(&entry.value, upper);
                if scan_node.upper_inclusive {
                    if cmp == std::cmp::Ordering::Greater {
                        continue; // Skip values > upper
                    }
                } else if cmp != std::cmp::Ordering::Less {
                    continue; // Skip values >= upper
                }
            }

            if let Some(entity) = tx.get_entity(entry.entity_id).map_err(Error::Transaction)? {
                entities.push(entity);
            }
        }
    }

    Ok(entities)
}

/// Compare two values for ordering.
fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Less,
        (_, Value::Null) => Ordering::Greater,
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        _ => Ordering::Equal, // Incomparable types are considered equal
    }
}

/// Execute an IN list lookup using the index.
///
/// Performs multiple point lookups for each value in the IN list
/// and combines the results.
pub fn execute_index_in_scan<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    table_name: &str,
    column_name: &str,
    values: &[Value],
) -> Result<Vec<Entity>> {
    let index_id = IndexId::from_label_property(table_name, column_name);

    let mut entity_ids = std::collections::HashSet::new();
    let mut entities = Vec::new();

    for value in values {
        // Get the scan range for this exact value
        if let Some((start, end)) = PropertyIndexScan::exact_value_range(index_id, value) {
            // Scan the property index
            let keys = tx.scan_property_index(&start, &end).map_err(Error::Transaction)?;

            for key in keys {
                if let Some(entry) = PropertyIndexEntry::decode_key(&key) {
                    // Deduplicate entities
                    if entity_ids.insert(entry.entity_id) {
                        if let Some(entity) =
                            tx.get_entity(entry.entity_id).map_err(Error::Transaction)?
                        {
                            entities.push(entity);
                        }
                    }
                }
            }
        }
    }

    Ok(entities)
}

/// Execute a prefix scan using the index.
///
/// Finds all entities where the indexed string column starts with
/// the given prefix (for LIKE 'prefix%' queries).
pub fn execute_index_prefix_scan<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    table_name: &str,
    column_name: &str,
    prefix: &str,
) -> Result<Vec<Entity>> {
    let index_id = IndexId::from_label_property(table_name, column_name);

    // Get the scan range for this prefix
    let (start, end) = PropertyIndexScan::string_prefix_range(index_id, prefix)
        .ok_or_else(|| Error::Execution("Cannot create prefix scan".to_string()))?;

    // Scan the property index
    let keys = tx.scan_property_index(&start, &end).map_err(Error::Transaction)?;

    // Decode entity IDs and fetch entities
    let mut entities = Vec::with_capacity(keys.len());
    for key in keys {
        if let Some(entry) = PropertyIndexEntry::decode_key(&key) {
            if let Some(entity) = tx.get_entity(entry.entity_id).map_err(Error::Transaction)? {
                entities.push(entity);
            }
        }
    }

    Ok(entities)
}

#[cfg(test)]
mod tests {
    // Tests will be in integration tests since they need a real database
}

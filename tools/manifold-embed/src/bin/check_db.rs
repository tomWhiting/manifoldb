//! Quick utility to check database contents

use manifoldb::DatabaseBuilder;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let db_path = args.get(1).map(|s| s.as_str()).unwrap_or("embeddings.manifold");

    println!("Opening database: {}", db_path);
    let db = DatabaseBuilder::new()
        .path(db_path)
        .open()?;

    // Read an entity directly by ID
    println!("\n=== Get entity by ID ===");
    let tx = db.begin_read()?;
    if let Some(entity) = tx.get_entity(manifoldb::EntityId::new(1))? {
        println!("  Entity 1: {:?}", entity);
    } else {
        println!("  Entity 1: not found");
    }
    if let Some(entity) = tx.get_entity(manifoldb::EntityId::new(3))? {
        println!("  Entity 3: {:?}", entity);
    } else {
        println!("  Entity 3: not found");
    }
    drop(tx);

    // Try SQL query for chunks
    println!("\n=== SQL query chunks (Chunk label) ===");
    match db.query("SELECT _rowid, heading, file_path FROM Chunk LIMIT 5") {
        Ok(rows) => {
            if rows.is_empty() {
                println!("  (empty)");
            }
            for row in rows {
                println!("  {:?}", row);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }

    // Check edge count
    println!("\n=== Edge count ===");
    match db.query("SELECT COUNT(*) FROM edges") {
        Ok(rows) => {
            for row in rows {
                println!("  {:?}", row);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }

    // Check for embedding columns - try old format
    println!("\n=== Check embedding column (old format) ===");
    match db.query("SELECT _rowid FROM Chunk WHERE embedding IS NOT NULL LIMIT 1") {
        Ok(rows) => {
            if rows.is_empty() {
                println!("  No chunks with 'embedding' column");
            } else {
                println!("  Found chunks with 'embedding' column");
            }
        }
        Err(e) => println!("  Error: {}", e),
    }

    // Check for embedding_dense column - new format
    println!("\n=== Check embedding column (new format) ===");
    match db.query("SELECT _rowid FROM Chunk WHERE embedding_dense IS NOT NULL LIMIT 1") {
        Ok(rows) => {
            if rows.is_empty() {
                println!("  No chunks with 'embedding_dense' column");
            } else {
                println!("  Found chunks with 'embedding_dense' column");
            }
        }
        Err(e) => println!("  Error: {}", e),
    }

    // Count total chunks
    println!("\n=== Chunk count ===");
    match db.query("SELECT COUNT(*) FROM Chunk") {
        Ok(rows) => {
            for row in rows {
                println!("  {:?}", row);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }

    // Test vector distance query with explicit embedding column
    println!("\n=== Test vector distance query (embedding column) ===");
    // Get a sample embedding first
    let sample_query = "SELECT _rowid, embedding FROM Chunk LIMIT 1";
    match db.query(sample_query) {
        Ok(rows) => {
            if let Some(row) = rows.into_iter().next() {
                println!("  Sample row values ({} items):", row.values().len());
                for (i, v) in row.values().iter().enumerate() {
                    let type_name = match v {
                        manifoldb::Value::Null => "Null",
                        manifoldb::Value::Bool(_) => "Bool",
                        manifoldb::Value::Int(_) => "Int",
                        manifoldb::Value::Float(_) => "Float",
                        manifoldb::Value::String(_) => "String",
                        manifoldb::Value::Vector(_) => "Vector",
                        _ => "Other",
                    };
                    println!("    [{}]: {} - {:?}", i, type_name, if matches!(v, manifoldb::Value::Vector(_)) {
                        "Vector(...)".to_string()
                    } else {
                        format!("{:?}", v).chars().take(100).collect::<String>()
                    });
                }
            }
        }
        Err(e) => println!("  Error: {}", e),
    }

    // Test distance query - same pattern as search
    println!("\n=== Test distance query ===");
    // Get first embedding and use it to query
    match db.query("SELECT embedding FROM Chunk LIMIT 1") {
        Ok(rows) => {
            if let Some(row) = rows.into_iter().next() {
                if let Some(manifoldb::Value::Vector(vec)) = row.values().first() {
                    println!("  Got embedding with {} dimensions", vec.len());
                    // Now do distance query
                    let query = "SELECT _rowid, embedding <-> $1 AS distance FROM Chunk ORDER BY embedding <-> $1 LIMIT 3";
                    match db.query_with_params(query, &[manifoldb::Value::Vector(vec.clone())]) {
                        Ok(results) => {
                            println!("  Distance query returned {} rows", results.len());
                            for (i, row) in results.into_iter().enumerate() {
                                println!("    Row {}: {} values", i, row.values().len());
                                for (j, v) in row.values().iter().enumerate() {
                                    let type_name = match v {
                                        manifoldb::Value::Null => "Null",
                                        manifoldb::Value::Bool(_) => "Bool",
                                        manifoldb::Value::Int(_) => "Int",
                                        manifoldb::Value::Float(f) => {
                                            println!("      [{}]: Float({})", j, f);
                                            continue;
                                        }
                                        manifoldb::Value::String(_) => "String",
                                        manifoldb::Value::Vector(_) => "Vector",
                                        _ => "Other",
                                    };
                                    println!("      [{}]: {}", j, type_name);
                                }
                            }
                        }
                        Err(e) => println!("  Distance query error: {}", e),
                    }
                }
            }
        }
        Err(e) => println!("  Error: {}", e),
    }

    Ok(())
}

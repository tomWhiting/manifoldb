//! Basic usage example for ManifoldDB.
//!
//! This example demonstrates:
//! - Opening an in-memory database
//! - Creating entities with labels and properties
//! - Storing and retrieving entities
//! - Using transactions for atomic operations
//!
//! Run with: `cargo run --example basic_usage`

use manifoldb::{Database, Error, Value};

fn main() -> Result<(), Error> {
    // Create an in-memory database for this example
    // In production, use Database::open("path/to/db.manifold")
    let db = Database::in_memory()?;

    println!("ManifoldDB Basic Usage Example");
    println!("==============================\n");

    // Create some entities in a transaction
    let alice_id = {
        let mut tx = db.begin()?;

        // Create a person entity with labels and properties
        let alice = tx
            .create_entity()?
            .with_label("Person")
            .with_label("Employee")
            .with_property("name", "Alice")
            .with_property("age", 30i64)
            .with_property("email", "alice@example.com");

        let alice_id = alice.id;
        tx.put_entity(&alice)?;

        // Create another person
        let bob = tx
            .create_entity()?
            .with_label("Person")
            .with_property("name", "Bob")
            .with_property("age", 25i64);

        tx.put_entity(&bob)?;

        // Commit the transaction
        tx.commit()?;

        println!("Created Alice (id: {:?}) and Bob", alice_id);
        alice_id
    };

    // Read entities in a separate transaction
    {
        let tx = db.begin_read()?;

        // Get Alice by ID
        if let Some(alice) = tx.get_entity(alice_id)? {
            println!("\nRetrieved Alice:");
            println!("  ID: {:?}", alice.id);
            println!("  Labels: {:?}", alice.labels.iter().collect::<Vec<_>>());

            if let Some(Value::String(name)) = alice.get_property("name") {
                println!("  Name: {}", name);
            }

            if let Some(Value::Int(age)) = alice.get_property("age") {
                println!("  Age: {}", age);
            }
        }
    }

    // Update an entity
    {
        let mut tx = db.begin()?;

        if let Some(mut alice) = tx.get_entity(alice_id)? {
            // Update the age property
            alice.set_property("age", 31i64);
            alice.set_property("department", "Engineering");

            tx.put_entity(&alice)?;
            tx.commit()?;

            println!("\nUpdated Alice's age to 31 and added department");
        }
    }

    // Verify the update
    {
        let tx = db.begin_read()?;

        if let Some(alice) = tx.get_entity(alice_id)? {
            println!("\nAfter update:");
            if let Some(Value::Int(age)) = alice.get_property("age") {
                println!("  Age: {}", age);
            }
            if let Some(Value::String(dept)) = alice.get_property("department") {
                println!("  Department: {}", dept);
            }
        }
    }

    // Demonstrate rollback
    {
        let mut tx = db.begin()?;

        if let Some(mut alice) = tx.get_entity(alice_id)? {
            alice.set_property("age", 999i64);
            tx.put_entity(&alice)?;
            // Rollback - changes will NOT be persisted
            tx.rollback()?;
            println!("\nRolled back changes (age would have been 999)");
        }
    }

    // Verify rollback worked
    {
        let tx = db.begin_read()?;

        if let Some(alice) = tx.get_entity(alice_id)? {
            if let Some(Value::Int(age)) = alice.get_property("age") {
                println!("After rollback, age is still: {}", age);
            }
        }
    }

    // Delete an entity
    {
        let mut tx = db.begin()?;

        // Create a temporary entity
        let temp = tx.create_entity()?.with_label("Temporary");
        let temp_id = temp.id;
        tx.put_entity(&temp)?;
        tx.commit()?;

        // Now delete it
        let mut tx = db.begin()?;
        let deleted = tx.delete_entity(temp_id)?;
        tx.commit()?;

        println!("\nDeleted temporary entity: {}", deleted);
    }

    println!("\nBasic usage example complete!");

    Ok(())
}

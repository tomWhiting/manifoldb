//! Graph queries example for ManifoldDB.
//!
//! This example demonstrates:
//! - Creating graph structures with nodes and edges
//! - Traversing relationships
//! - Using different edge types
//!
//! Run with: `cargo run --example graph_queries`

use manifoldb::{Database, Error, Value};

fn main() -> Result<(), Error> {
    let db = Database::in_memory()?;

    println!("ManifoldDB Graph Queries Example");
    println!("=================================\n");

    // Build a social network graph
    let (alice_id, bob_id, carol_id, dave_id) = {
        let mut tx = db.begin()?;

        // Create people
        let alice = tx.create_entity()?.with_label("Person").with_property("name", "Alice");
        let bob = tx.create_entity()?.with_label("Person").with_property("name", "Bob");
        let carol = tx.create_entity()?.with_label("Person").with_property("name", "Carol");
        let dave = tx.create_entity()?.with_label("Person").with_property("name", "Dave");

        let (alice_id, bob_id, carol_id, dave_id) = (alice.id, bob.id, carol.id, dave.id);

        tx.put_entity(&alice)?;
        tx.put_entity(&bob)?;
        tx.put_entity(&carol)?;
        tx.put_entity(&dave)?;

        // Create FOLLOWS relationships
        // Alice -> Bob, Alice -> Carol
        // Bob -> Carol
        // Carol -> Dave
        // Dave -> Alice (circular!)

        let e1 = tx.create_edge(alice_id, bob_id, "FOLLOWS")?.with_property("since", "2023");
        let e2 = tx.create_edge(alice_id, carol_id, "FOLLOWS")?.with_property("since", "2022");
        let e3 = tx.create_edge(bob_id, carol_id, "FOLLOWS")?.with_property("since", "2024");
        let e4 = tx.create_edge(carol_id, dave_id, "FOLLOWS")?.with_property("since", "2021");
        let e5 = tx.create_edge(dave_id, alice_id, "FOLLOWS")?.with_property("since", "2020");

        tx.put_edge(&e1)?;
        tx.put_edge(&e2)?;
        tx.put_edge(&e3)?;
        tx.put_edge(&e4)?;
        tx.put_edge(&e5)?;

        // Create WORKS_WITH relationships
        let w1 = tx.create_edge(alice_id, bob_id, "WORKS_WITH")?;
        let w2 = tx.create_edge(carol_id, dave_id, "WORKS_WITH")?;

        tx.put_edge(&w1)?;
        tx.put_edge(&w2)?;

        tx.commit()?;

        println!("Created social network with 4 people and relationships");
        (alice_id, bob_id, carol_id, dave_id)
    };

    // Get a person's name by ID
    let get_name = |db: &Database, id| -> Result<String, Error> {
        let tx = db.begin_read()?;
        if let Some(entity) = tx.get_entity(id)? {
            if let Some(Value::String(name)) = entity.get_property("name") {
                return Ok(name.clone());
            }
        }
        Ok("Unknown".to_string())
    };

    // Traverse outgoing edges (who does Alice follow?)
    {
        let tx = db.begin_read()?;

        println!("\n{} follows:", get_name(&db, alice_id)?);
        let edges = tx.get_outgoing_edges(alice_id)?;
        for edge in &edges {
            if edge.edge_type.as_str() == "FOLLOWS" {
                let name = get_name(&db, edge.target)?;
                if let Some(Value::String(since)) = edge.get_property("since") {
                    println!("  -> {} (since {})", name, since);
                } else {
                    println!("  -> {}", name);
                }
            }
        }
    }

    // Traverse incoming edges (who follows Carol?)
    {
        let tx = db.begin_read()?;

        println!("\n{} is followed by:", get_name(&db, carol_id)?);
        let edges = tx.get_incoming_edges(carol_id)?;
        for edge in &edges {
            if edge.edge_type.as_str() == "FOLLOWS" {
                let name = get_name(&db, edge.source)?;
                println!("  <- {}", name);
            }
        }
    }

    // Multi-hop traversal (friends of friends)
    {
        let tx = db.begin_read()?;

        println!("\nAlice's friends of friends (2-hop):");
        let mut visited = std::collections::HashSet::new();
        visited.insert(alice_id);

        // First hop
        let first_hop = tx.get_outgoing_edges(alice_id)?;
        for edge in &first_hop {
            if edge.edge_type.as_str() == "FOLLOWS" {
                visited.insert(edge.target);

                // Second hop
                let second_hop = tx.get_outgoing_edges(edge.target)?;
                for edge2 in &second_hop {
                    if edge2.edge_type.as_str() == "FOLLOWS" && !visited.contains(&edge2.target) {
                        let name = get_name(&db, edge2.target)?;
                        let via = get_name(&db, edge.target)?;
                        println!("  {} (via {})", name, via);
                        visited.insert(edge2.target);
                    }
                }
            }
        }
    }

    // Filter by edge type
    {
        let tx = db.begin_read()?;

        println!("\nAlice's coworkers (WORKS_WITH edges):");
        let edges = tx.get_outgoing_edges(alice_id)?;
        for edge in &edges {
            if edge.edge_type.as_str() == "WORKS_WITH" {
                let name = get_name(&db, edge.target)?;
                println!("  - {}", name);
            }
        }
    }

    // Count relationships
    {
        let tx = db.begin_read()?;

        println!("\nRelationship counts:");
        for (id, name) in
            [(alice_id, "Alice"), (bob_id, "Bob"), (carol_id, "Carol"), (dave_id, "Dave")]
        {
            let outgoing = tx.get_outgoing_edges(id)?;
            let incoming = tx.get_incoming_edges(id)?;
            let follows_out = outgoing.iter().filter(|e| e.edge_type.as_str() == "FOLLOWS").count();
            let follows_in = incoming.iter().filter(|e| e.edge_type.as_str() == "FOLLOWS").count();
            println!("  {}: follows {} | followed by {}", name, follows_out, follows_in);
        }
    }

    // Delete an edge
    {
        let mut tx = db.begin()?;

        // Find and delete Alice -> Bob FOLLOWS edge
        let edges = tx.get_outgoing_edges(alice_id)?;
        for edge in &edges {
            if edge.edge_type.as_str() == "FOLLOWS" && edge.target == bob_id {
                tx.delete_edge(edge.id)?;
                println!("\nDeleted Alice -> Bob FOLLOWS relationship");
                break;
            }
        }
        tx.commit()?;
    }

    // Verify deletion
    {
        let tx = db.begin_read()?;

        let edges = tx.get_outgoing_edges(alice_id)?;
        let follows_bob =
            edges.iter().any(|e| e.edge_type.as_str() == "FOLLOWS" && e.target == bob_id);

        println!("Alice still follows Bob: {}", follows_bob);
    }

    println!("\nGraph queries example complete!");

    Ok(())
}

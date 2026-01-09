//! Procedure registry for managing callable procedures.

use std::collections::HashMap;
use std::sync::Arc;

use super::signature::ProcedureSignature;
use super::traits::{Procedure, ProcedureError, ProcedureResult};

/// A registry of callable procedures.
///
/// The registry stores procedures by their fully-qualified names (e.g., "algo.pageRank")
/// and provides lookup, registration, and listing functionality.
///
/// # Example
///
/// ```ignore
/// let mut registry = ProcedureRegistry::new();
/// registry.register(Arc::new(PageRankProcedure));
///
/// if let Some(proc) = registry.get("algo.pageRank") {
///     let result = proc.execute(args)?;
/// }
/// ```
#[derive(Default)]
pub struct ProcedureRegistry {
    /// Map of procedure names to procedure implementations.
    procedures: HashMap<String, Arc<dyn Procedure>>,
}

impl ProcedureRegistry {
    /// Creates a new empty procedure registry.
    #[must_use]
    pub fn new() -> Self {
        Self { procedures: HashMap::new() }
    }

    /// Registers a procedure.
    ///
    /// If a procedure with the same name already exists, it is replaced.
    pub fn register(&mut self, procedure: Arc<dyn Procedure>) {
        let name = procedure.signature().name.clone();
        self.procedures.insert(name, procedure);
    }

    /// Gets a procedure by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn Procedure>> {
        self.procedures.get(name).cloned()
    }

    /// Gets a procedure by name, returning an error if not found.
    pub fn get_or_error(&self, name: &str) -> ProcedureResult<Arc<dyn Procedure>> {
        self.get(name).ok_or_else(|| ProcedureError::NotFound(name.to_string()))
    }

    /// Returns true if a procedure with the given name exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.procedures.contains_key(name)
    }

    /// Removes a procedure from the registry.
    ///
    /// Returns the removed procedure if it existed.
    pub fn unregister(&mut self, name: &str) -> Option<Arc<dyn Procedure>> {
        self.procedures.remove(name)
    }

    /// Returns the number of registered procedures.
    #[must_use]
    pub fn len(&self) -> usize {
        self.procedures.len()
    }

    /// Returns true if no procedures are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.procedures.is_empty()
    }

    /// Lists all registered procedure names.
    #[must_use]
    pub fn list_names(&self) -> Vec<&str> {
        self.procedures.keys().map(String::as_str).collect()
    }

    /// Lists all registered procedure signatures.
    #[must_use]
    pub fn list_signatures(&self) -> Vec<ProcedureSignature> {
        self.procedures.values().map(|p| p.signature()).collect()
    }

    /// Returns an iterator over all registered procedures.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Arc<dyn Procedure>)> {
        self.procedures.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Merges another registry into this one.
    ///
    /// Procedures from the other registry will overwrite any with the same name.
    pub fn merge(&mut self, other: ProcedureRegistry) {
        self.procedures.extend(other.procedures);
    }
}

impl std::fmt::Debug for ProcedureRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcedureRegistry").field("procedures", &self.list_names()).finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use manifoldb_core::Value;

    use super::*;
    use crate::exec::{RowBatch, Schema};
    use crate::procedure::{ProcedureArgs, ProcedureParameter, ReturnColumn};

    // Test procedure implementation
    struct TestProcedure;

    impl Procedure for TestProcedure {
        fn signature(&self) -> ProcedureSignature {
            ProcedureSignature::new("test.echo")
                .with_description("Returns the input value")
                .with_parameter(ProcedureParameter::required("value", "STRING"))
                .with_return(ReturnColumn::new("result", "STRING"))
        }

        fn execute(&self, args: ProcedureArgs) -> ProcedureResult<RowBatch> {
            let schema = Arc::new(Schema::new(vec!["result".to_string()]));
            let mut batch = RowBatch::new(Arc::clone(&schema));

            if let Some(value) = args.get(0) {
                batch.push(crate::exec::Row::new(schema, vec![value.clone()]));
            }

            Ok(batch)
        }
    }

    #[test]
    fn registry_register_and_get() {
        let mut registry = ProcedureRegistry::new();
        registry.register(Arc::new(TestProcedure));

        assert!(registry.contains("test.echo"));
        assert!(!registry.contains("test.unknown"));

        let proc = registry.get("test.echo");
        assert!(proc.is_some());
        assert_eq!(
            proc.as_ref().map(|p| p.signature().name.clone()),
            Some("test.echo".to_string())
        );
    }

    #[test]
    fn registry_execute() {
        let mut registry = ProcedureRegistry::new();
        registry.register(Arc::new(TestProcedure));

        let proc = registry.get("test.echo").expect("procedure should exist");
        let args = ProcedureArgs::new(vec![Value::from("hello")]);
        let result = proc.execute(args).expect("execution should succeed");

        assert_eq!(result.len(), 1);
        assert_eq!(result.rows()[0].get(0), Some(&Value::from("hello")));
    }

    #[test]
    fn registry_list() {
        let mut registry = ProcedureRegistry::new();
        registry.register(Arc::new(TestProcedure));

        let names = registry.list_names();
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"test.echo"));
    }

    #[test]
    fn registry_unregister() {
        let mut registry = ProcedureRegistry::new();
        registry.register(Arc::new(TestProcedure));

        assert!(registry.contains("test.echo"));
        registry.unregister("test.echo");
        assert!(!registry.contains("test.echo"));
    }
}

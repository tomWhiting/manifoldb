//! Procedure signatures for parameter and return type definitions.

/// A parameter definition for a procedure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcedureParameter {
    /// The parameter name.
    pub name: String,
    /// A description of the parameter.
    pub description: String,
    /// Whether this parameter is required.
    pub required: bool,
    /// The expected type of the parameter (for documentation).
    pub type_hint: String,
}

impl ProcedureParameter {
    /// Creates a required parameter.
    #[must_use]
    pub fn required(name: impl Into<String>, type_hint: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            required: true,
            type_hint: type_hint.into(),
        }
    }

    /// Creates an optional parameter.
    #[must_use]
    pub fn optional(name: impl Into<String>, type_hint: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            required: false,
            type_hint: type_hint.into(),
        }
    }

    /// Sets the description for this parameter.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

/// A column definition returned by a procedure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReturnColumn {
    /// The column name.
    pub name: String,
    /// A description of the column.
    pub description: String,
    /// The type of the column (for documentation).
    pub type_hint: String,
}

impl ReturnColumn {
    /// Creates a new return column definition.
    #[must_use]
    pub fn new(name: impl Into<String>, type_hint: impl Into<String>) -> Self {
        Self { name: name.into(), description: String::new(), type_hint: type_hint.into() }
    }

    /// Sets the description for this column.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

/// The signature of a procedure, including name, parameters, and return columns.
#[derive(Debug, Clone)]
pub struct ProcedureSignature {
    /// The fully-qualified procedure name (e.g., "algo.pageRank").
    pub name: String,
    /// A description of what the procedure does.
    pub description: String,
    /// The parameters the procedure accepts.
    pub parameters: Vec<ProcedureParameter>,
    /// The columns the procedure returns.
    pub returns: Vec<ReturnColumn>,
}

impl ProcedureSignature {
    /// Creates a new procedure signature.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), description: String::new(), parameters: vec![], returns: vec![] }
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Adds a parameter.
    #[must_use]
    pub fn with_parameter(mut self, param: ProcedureParameter) -> Self {
        self.parameters.push(param);
        self
    }

    /// Adds multiple parameters.
    #[must_use]
    pub fn with_parameters(mut self, params: Vec<ProcedureParameter>) -> Self {
        self.parameters.extend(params);
        self
    }

    /// Adds a return column.
    #[must_use]
    pub fn with_return(mut self, col: ReturnColumn) -> Self {
        self.returns.push(col);
        self
    }

    /// Adds multiple return columns.
    #[must_use]
    pub fn with_returns(mut self, cols: Vec<ReturnColumn>) -> Self {
        self.returns.extend(cols);
        self
    }

    /// Returns the names of all return columns.
    #[must_use]
    pub fn return_column_names(&self) -> Vec<&str> {
        self.returns.iter().map(|c| c.name.as_str()).collect()
    }

    /// Returns the number of required parameters.
    #[must_use]
    pub fn required_param_count(&self) -> usize {
        self.parameters.iter().filter(|p| p.required).count()
    }

    /// Validates the number of arguments.
    pub fn validate_arg_count(&self, arg_count: usize) -> Result<(), String> {
        let min = self.required_param_count();
        let max = self.parameters.len();

        if arg_count < min {
            return Err(format!(
                "procedure '{}' requires at least {} argument(s), got {}",
                self.name, min, arg_count
            ));
        }

        if arg_count > max {
            return Err(format!(
                "procedure '{}' accepts at most {} argument(s), got {}",
                self.name, max, arg_count
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_builder() {
        let sig = ProcedureSignature::new("algo.pageRank")
            .with_description("Computes PageRank scores")
            .with_parameter(
                ProcedureParameter::required("node_label", "STRING")
                    .with_description("Label of nodes to include"),
            )
            .with_parameter(
                ProcedureParameter::required("rel_type", "STRING")
                    .with_description("Relationship type for edges"),
            )
            .with_parameter(
                ProcedureParameter::optional("damping", "FLOAT")
                    .with_description("Damping factor (default 0.85)"),
            )
            .with_return(ReturnColumn::new("node", "NODE").with_description("The node"))
            .with_return(ReturnColumn::new("score", "FLOAT").with_description("PageRank score"));

        assert_eq!(sig.name, "algo.pageRank");
        assert_eq!(sig.parameters.len(), 3);
        assert_eq!(sig.returns.len(), 2);
        assert_eq!(sig.required_param_count(), 2);
    }

    #[test]
    fn validate_arg_count() {
        let sig = ProcedureSignature::new("test")
            .with_parameter(ProcedureParameter::required("a", "STRING"))
            .with_parameter(ProcedureParameter::required("b", "STRING"))
            .with_parameter(ProcedureParameter::optional("c", "STRING"));

        assert!(sig.validate_arg_count(2).is_ok());
        assert!(sig.validate_arg_count(3).is_ok());
        assert!(sig.validate_arg_count(1).is_err());
        assert!(sig.validate_arg_count(4).is_err());
    }
}

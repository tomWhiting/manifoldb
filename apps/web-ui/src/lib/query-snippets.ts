export type SnippetCategory = 'cypher' | 'sql'

export interface QuerySnippet {
  id: string
  name: string
  description: string
  query: string
  category: SnippetCategory
}

export interface QueryTemplate {
  id: string
  name: string
  description: string
  template: string
  placeholders: TemplatePlaceholder[]
  category: SnippetCategory
}

export interface TemplatePlaceholder {
  key: string
  label: string
  defaultValue: string
}

export const cypherSnippets: QuerySnippet[] = [
  {
    id: 'cypher-all-nodes',
    name: 'Get all nodes',
    description: 'Retrieve all nodes with a limit',
    query: 'MATCH (n) RETURN n LIMIT 10',
    category: 'cypher',
  },
  {
    id: 'cypher-relationships',
    name: 'Get relationships',
    description: 'Retrieve nodes and their relationships',
    query: 'MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25',
    category: 'cypher',
  },
  {
    id: 'cypher-by-label',
    name: 'Get by label',
    description: 'Retrieve nodes with a specific label',
    query: 'MATCH (n:Label) RETURN n',
    category: 'cypher',
  },
  {
    id: 'cypher-create-node',
    name: 'Create node',
    description: 'Create a new node with properties',
    query: 'CREATE (n:Label {prop: value})',
    category: 'cypher',
  },
  {
    id: 'cypher-create-relationship',
    name: 'Create relationship',
    description: 'Create a relationship between nodes',
    query: 'MATCH (a), (b) CREATE (a)-[:REL]->(b)',
    category: 'cypher',
  },
  {
    id: 'cypher-count-nodes',
    name: 'Count nodes',
    description: 'Count all nodes in the database',
    query: 'MATCH (n) RETURN count(n) AS nodeCount',
    category: 'cypher',
  },
  {
    id: 'cypher-count-by-label',
    name: 'Count by label',
    description: 'Count nodes grouped by label',
    query: 'MATCH (n) RETURN labels(n) AS label, count(*) AS count',
    category: 'cypher',
  },
  {
    id: 'cypher-delete-node',
    name: 'Delete node',
    description: 'Delete a node and its relationships',
    query: 'MATCH (n:Label {prop: value}) DETACH DELETE n',
    category: 'cypher',
  },
]

export const sqlSnippets: QuerySnippet[] = [
  {
    id: 'sql-all-nodes',
    name: 'Select all nodes',
    description: 'Retrieve all nodes from the nodes table',
    query: 'SELECT * FROM nodes LIMIT 10',
    category: 'sql',
  },
  {
    id: 'sql-all-edges',
    name: 'Select all edges',
    description: 'Retrieve all edges from the edges table',
    query: 'SELECT * FROM edges LIMIT 10',
    category: 'sql',
  },
  {
    id: 'sql-count-nodes',
    name: 'Count nodes',
    description: 'Count all nodes',
    query: 'SELECT COUNT(*) AS node_count FROM nodes',
    category: 'sql',
  },
  {
    id: 'sql-count-edges',
    name: 'Count edges',
    description: 'Count all edges',
    query: 'SELECT COUNT(*) AS edge_count FROM edges',
    category: 'sql',
  },
]

export const allSnippets: QuerySnippet[] = [...cypherSnippets, ...sqlSnippets]

export const cypherTemplates: QueryTemplate[] = [
  {
    id: 'template-find-by-property',
    name: 'Find by property',
    description: 'Find nodes with a specific property value',
    template: 'MATCH (n:${label}) WHERE n.${property} = ${value} RETURN n LIMIT ${limit}',
    placeholders: [
      { key: 'label', label: 'Label', defaultValue: 'Person' },
      { key: 'property', label: 'Property', defaultValue: 'name' },
      { key: 'value', label: 'Value', defaultValue: '"John"' },
      { key: 'limit', label: 'Limit', defaultValue: '10' },
    ],
    category: 'cypher',
  },
  {
    id: 'template-find-connected',
    name: 'Find connected nodes',
    description: 'Find nodes connected to a specific node',
    template:
      'MATCH (source:${sourceLabel} {${sourceProperty}: ${sourceValue}})-[r:${relType}]->(target)\nRETURN source, r, target LIMIT ${limit}',
    placeholders: [
      { key: 'sourceLabel', label: 'Source Label', defaultValue: 'Person' },
      { key: 'sourceProperty', label: 'Source Property', defaultValue: 'name' },
      { key: 'sourceValue', label: 'Source Value', defaultValue: '"John"' },
      { key: 'relType', label: 'Relationship Type', defaultValue: 'KNOWS' },
      { key: 'limit', label: 'Limit', defaultValue: '25' },
    ],
    category: 'cypher',
  },
  {
    id: 'template-shortest-path',
    name: 'Shortest path',
    description: 'Find shortest path between two nodes',
    template:
      'MATCH (a:${labelA} {${propA}: ${valueA}}), (b:${labelB} {${propB}: ${valueB}})\nMATCH path = shortestPath((a)-[*]-(b))\nRETURN path',
    placeholders: [
      { key: 'labelA', label: 'Start Label', defaultValue: 'Person' },
      { key: 'propA', label: 'Start Property', defaultValue: 'name' },
      { key: 'valueA', label: 'Start Value', defaultValue: '"Alice"' },
      { key: 'labelB', label: 'End Label', defaultValue: 'Person' },
      { key: 'propB', label: 'End Property', defaultValue: 'name' },
      { key: 'valueB', label: 'End Value', defaultValue: '"Bob"' },
    ],
    category: 'cypher',
  },
  {
    id: 'template-update-property',
    name: 'Update property',
    description: 'Update a property on matching nodes',
    template: 'MATCH (n:${label}) WHERE n.${matchProp} = ${matchValue}\nSET n.${setProp} = ${setValue}\nRETURN n',
    placeholders: [
      { key: 'label', label: 'Label', defaultValue: 'Person' },
      { key: 'matchProp', label: 'Match Property', defaultValue: 'name' },
      { key: 'matchValue', label: 'Match Value', defaultValue: '"John"' },
      { key: 'setProp', label: 'Set Property', defaultValue: 'age' },
      { key: 'setValue', label: 'Set Value', defaultValue: '30' },
    ],
    category: 'cypher',
  },
]

export const sqlTemplates: QueryTemplate[] = [
  {
    id: 'template-sql-filter',
    name: 'Filter nodes',
    description: 'Filter nodes by property value',
    template: 'SELECT * FROM nodes WHERE properties->>\'${property}\' = \'${value}\' LIMIT ${limit}',
    placeholders: [
      { key: 'property', label: 'Property', defaultValue: 'name' },
      { key: 'value', label: 'Value', defaultValue: 'John' },
      { key: 'limit', label: 'Limit', defaultValue: '10' },
    ],
    category: 'sql',
  },
]

export const allTemplates: QueryTemplate[] = [...cypherTemplates, ...sqlTemplates]

/** Apply placeholder values to a template string */
export function applyTemplate(
  template: string,
  placeholders: TemplatePlaceholder[],
  values?: Record<string, string>
): string {
  let result = template
  for (const placeholder of placeholders) {
    const value = values?.[placeholder.key] ?? placeholder.defaultValue
    result = result.replace(new RegExp(`\\$\\{${placeholder.key}\\}`, 'g'), value)
  }
  return result
}

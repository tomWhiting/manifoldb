# Schema Panel Review

**Task:** Schema panel with node/edge type browser
**Branch:** vk/14b9-schema-panel-wit
**Date:** 2026-01-11

---

## 1. Summary

Reviewed the implementation of the Schema sidebar panel for the web UI. The panel displays graph schema information including node labels and relationship types with counts, supports refresh functionality, and enables sample query generation on click.

## 2. Files Changed

### Created
- `apps/web-ui/src/hooks/useSchema.ts` - React hook for fetching and caching schema data
- `apps/web-ui/src/components/sidebar/SchemaPanel.tsx` - Schema panel UI component

### Modified
- `apps/web-ui/src/components/layout/Workspace.tsx` - Integrated SchemaPanel to render when 'schema' sidebar section is active

## 3. Issues Found

### Issue 1: Cypher Identifier Escaping (Fixed)
**Location:** `apps/web-ui/src/components/sidebar/SchemaPanel.tsx:91-98`

**Problem:** The generated Cypher queries for sample data did not escape label and relationship type names. If a label or type contains special characters (spaces, hyphens, etc.), the Cypher query would be syntactically invalid.

**Example of problem:**
```typescript
// Original - would break for label "My Label"
const query = `MATCH (n:${label.name}) RETURN n LIMIT 10`
// Would generate: MATCH (n:My Label) RETURN n LIMIT 10  <- Invalid Cypher
```

**Fix applied:** Added `escapeCypherIdentifier()` function that wraps identifiers containing special characters in backticks, with proper escaping of embedded backticks.

### Issue 2: Missing Features (Not Fixed - Backend Limitation)
**Location:** Task description requirements

**Problem:** The task description mentions:
- "Show properties per label/type with data types"
- "Show constraints and indexes"

These features are not implemented. However, the backend GraphQL schema (`GraphStats` type in `crates/manifold-server/src/schema/types.rs:206-217`) does not expose property information, constraints, or indexes. This is a backend limitation, not an implementation oversight.

The acceptance criteria in the task only require:
- Labels display with counts
- Relationship types display with counts
- Click runs sample query
- `npm run build` passes

All acceptance criteria are met.

## 4. Changes Made

### Fix 1: Added Cypher Identifier Escaping

Added a utility function at `apps/web-ui/src/components/sidebar/SchemaPanel.tsx:78-85`:

```typescript
/** Escape a label or relationship type for use in Cypher queries */
function escapeCypherIdentifier(name: string): string {
  // If the name contains special characters, wrap in backticks
  // and escape any backticks within the name
  if (/[^a-zA-Z0-9_]/.test(name)) {
    return '`' + name.replace(/`/g, '``') + '`'
  }
  return name
}
```

Updated the click handlers at `apps/web-ui/src/components/sidebar/SchemaPanel.tsx:101-111`:

```typescript
const handleLabelClick = (label: LabelInfo) => {
  const escapedLabel = escapeCypherIdentifier(label.name)
  const query = `MATCH (n:${escapedLabel}) RETURN n LIMIT 10`
  runSampleQuery(query)
}

const handleEdgeTypeClick = (edgeType: EdgeTypeInfo) => {
  const escapedType = escapeCypherIdentifier(edgeType.name)
  const query = `MATCH ()-[r:${escapedType}]->() RETURN r LIMIT 10`
  runSampleQuery(query)
}
```

## 5. Test Results

### Build
```
> @manifoldb/web-ui@0.0.0 build
> tsc -b && vite build

vite v7.3.1 building client environment for production...
transforming...
✓ 1945 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   0.46 kB │ gzip:   0.30 kB
dist/assets/index-Cv3YB3Sq.css   19.68 kB │ gzip:   4.77 kB
dist/assets/index-DAbJqG4N.js   780.83 kB │ gzip: 249.79 kB
✓ built in 3.06s
```

### Lint
```
> @manifoldb/web-ui@0.0.0 lint
> eslint .
```
(No errors)

### TypeScript
```
> npx tsc --noEmit
```
(No errors)

## 6. Code Quality Assessment

### Error Handling
- Loading state properly handled with spinner
- Error state displays error message with retry button
- Empty states handled for no labels/types

### TypeScript
- TypeScript strict mode enabled and passing
- All types properly defined (LabelInfo, EdgeTypeInfo, GraphSchema, etc.)
- No `any` types used

### Component Structure
- Clean separation of concerns (useSchema hook vs SchemaPanel component)
- Reusable CollapsibleSection and SchemaItem subcomponents
- Follows existing patterns in the codebase (similar to useConnection hook)

### UI/UX
- Collapsible sections for Node Labels and Relationship Types
- Summary bar showing total node/edge counts
- Last updated timestamp in footer
- Refresh button with loading state

## 7. Verdict

**Approved with Fixes**

The implementation meets all acceptance criteria and follows the established patterns in the codebase. One issue was identified and fixed:
- Added proper Cypher identifier escaping for labels and relationship types containing special characters

The missing "properties per type" and "constraints/indexes" features mentioned in the task description are not available in the backend schema, so this is a backend limitation rather than an implementation gap.

import { useState, useCallback, useMemo } from 'react'
import {
  Upload,
  Download,
  FileJson,
  FileSpreadsheet,
  AlertCircle,
  CheckCircle,
  X,
  ChevronDown,
  Loader2,
} from 'lucide-react'
import { toast } from 'sonner'
import { CollapsibleSection } from '../shared/CollapsibleSection'
import { useWorkspaceStore } from '../../stores/workspace-store'
import {
  parseJsonImport,
  parseCsvImport,
  readFileAsText,
  detectCsvColumns,
  previewCsvData,
  type ImportProgress,
  type ImportResult,
  type CsvMapping,
} from '../../lib/import-utils'
import {
  exportQueryResultToJson,
  exportQueryResultToCsv,
  downloadFile,
  generateExportFilename,
  formatBytes,
  type ExportOptions,
} from '../../lib/export-utils'

type ImportStep = 'select' | 'configure' | 'progress' | 'result'

interface FileDropZoneProps {
  onFileSelect: (file: File) => void
  accept: string
  disabled?: boolean
}

function FileDropZone({ onFileSelect, accept, disabled }: FileDropZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    if (!disabled) {
      setIsDragOver(true)
    }
  }, [disabled])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    if (disabled) return

    const file = e.dataTransfer.files[0]
    if (file) {
      onFileSelect(file)
    }
  }, [disabled, onFileSelect])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onFileSelect(file)
    }
    e.target.value = ''
  }, [onFileSelect])

  return (
    <label
      className={`
        flex flex-col items-center justify-center
        w-full h-32 border-2 border-dashed rounded-lg
        transition-colors cursor-pointer
        ${isDragOver ? 'border-accent bg-accent/10' : 'border-border hover:border-accent/50'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <Upload size={24} className="text-text-muted mb-2" />
      <span className="text-sm text-text-muted">Drop files here or click to browse</span>
      <span className="text-xs text-text-muted mt-1">{accept}</span>
      <input
        type="file"
        accept={accept}
        onChange={handleFileInput}
        className="hidden"
        disabled={disabled}
      />
    </label>
  )
}

interface ImportProgressDisplayProps {
  progress: ImportProgress
}

function ImportProgressDisplay({ progress }: ImportProgressDisplayProps) {
  const percentage = progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-text-secondary capitalize">{progress.phase}</span>
        <span className="text-text-muted">{percentage}%</span>
      </div>
      <div className="w-full h-2 bg-bg-tertiary rounded-full overflow-hidden">
        <div
          className="h-full bg-accent transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <p className="text-xs text-text-muted">{progress.message}</p>
    </div>
  )
}

interface ImportResultDisplayProps {
  result: ImportResult
  onReset: () => void
}

function ImportResultDisplay({ result, onReset }: ImportResultDisplayProps) {
  const hasErrors = result.errors.length > 0
  const [showErrors, setShowErrors] = useState(false)

  return (
    <div className="space-y-4">
      <div className={`
        flex items-center gap-2 p-3 rounded-lg
        ${hasErrors ? 'bg-yellow-500/10 border border-yellow-500/20' : 'bg-green-500/10 border border-green-500/20'}
      `}>
        {hasErrors ? (
          <AlertCircle size={20} className="text-yellow-400 flex-shrink-0" />
        ) : (
          <CheckCircle size={20} className="text-green-400 flex-shrink-0" />
        )}
        <div className="flex-1">
          <p className={`text-sm font-medium ${hasErrors ? 'text-yellow-300' : 'text-green-300'}`}>
            {hasErrors ? 'Import completed with warnings' : 'Import successful'}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="bg-bg-tertiary rounded-lg p-3">
          <p className="text-text-muted text-xs">Nodes</p>
          <p className="text-text-primary font-medium text-lg">{result.stats.successfulNodes}</p>
        </div>
        <div className="bg-bg-tertiary rounded-lg p-3">
          <p className="text-text-muted text-xs">Edges</p>
          <p className="text-text-primary font-medium text-lg">{result.stats.successfulEdges}</p>
        </div>
      </div>

      {hasErrors && (
        <div>
          <button
            onClick={() => setShowErrors(!showErrors)}
            className="flex items-center gap-1 text-sm text-yellow-400 hover:text-yellow-300"
          >
            <ChevronDown size={14} className={showErrors ? 'rotate-180' : ''} />
            {result.errors.length} error{result.errors.length > 1 ? 's' : ''}
          </button>
          {showErrors && (
            <div className="mt-2 max-h-40 overflow-y-auto space-y-1">
              {result.errors.slice(0, 20).map((error, i) => (
                <div key={i} className="text-xs text-text-muted bg-bg-tertiary rounded px-2 py-1">
                  <span className="text-red-400">Line {error.line}:</span> {error.message}
                </div>
              ))}
              {result.errors.length > 20 && (
                <p className="text-xs text-text-muted">
                  ... and {result.errors.length - 20} more errors
                </p>
              )}
            </div>
          )}
        </div>
      )}

      <button
        onClick={onReset}
        className="w-full py-2 px-4 bg-bg-tertiary hover:bg-border text-text-secondary text-sm rounded transition-colors"
      >
        Import another file
      </button>
    </div>
  )
}

interface CsvConfigPanelProps {
  columns: string[]
  dataType: 'nodes' | 'edges'
  onDataTypeChange: (type: 'nodes' | 'edges') => void
  mapping: CsvMapping
  onMappingChange: (mapping: CsvMapping) => void
  preview: { headers: string[]; rows: string[][] }
  delimiter: string
  onDelimiterChange: (d: string) => void
}

function CsvConfigPanel({
  columns,
  dataType,
  onDataTypeChange,
  mapping,
  onMappingChange,
  preview,
  delimiter,
  onDelimiterChange,
}: CsvConfigPanelProps) {
  const updateMapping = (key: keyof CsvMapping, value: string | string[] | undefined) => {
    onMappingChange({ ...mapping, [key]: value || undefined })
  }

  return (
    <div className="space-y-4">
      {/* Data Type */}
      <div>
        <label className="block text-xs text-text-muted mb-1">Data Type</label>
        <div className="flex gap-2">
          <button
            onClick={() => onDataTypeChange('nodes')}
            className={`
              flex-1 py-1.5 px-3 text-sm rounded transition-colors
              ${dataType === 'nodes' ? 'bg-accent text-white' : 'bg-bg-tertiary text-text-secondary hover:bg-border'}
            `}
          >
            Nodes
          </button>
          <button
            onClick={() => onDataTypeChange('edges')}
            className={`
              flex-1 py-1.5 px-3 text-sm rounded transition-colors
              ${dataType === 'edges' ? 'bg-accent text-white' : 'bg-bg-tertiary text-text-secondary hover:bg-border'}
            `}
          >
            Edges
          </button>
        </div>
      </div>

      {/* Delimiter */}
      <div>
        <label className="block text-xs text-text-muted mb-1">Delimiter</label>
        <select
          value={delimiter}
          onChange={(e) => onDelimiterChange(e.target.value)}
          className="w-full px-2 py-1.5 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
        >
          <option value=",">Comma (,)</option>
          <option value=";">Semicolon (;)</option>
          <option value="\t">Tab</option>
          <option value="|">Pipe (|)</option>
        </select>
      </div>

      {/* Column Mapping */}
      <div className="space-y-2">
        <label className="block text-xs text-text-muted">Column Mapping</label>

        <div className="flex items-center gap-2">
          <span className="text-xs text-text-muted w-20">ID Column</span>
          <select
            value={mapping.idColumn}
            onChange={(e) => updateMapping('idColumn', e.target.value)}
            className="flex-1 px-2 py-1 text-xs bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
          >
            {columns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>

        {dataType === 'nodes' && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-muted w-20">Labels</span>
            <select
              value={mapping.labelsColumn ?? ''}
              onChange={(e) => updateMapping('labelsColumn', e.target.value)}
              className="flex-1 px-2 py-1 text-xs bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
            >
              <option value="">(none - use default)</option>
              {columns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>
        )}

        {dataType === 'edges' && (
          <>
            <div className="flex items-center gap-2">
              <span className="text-xs text-text-muted w-20">Type</span>
              <select
                value={mapping.typeColumn ?? ''}
                onChange={(e) => updateMapping('typeColumn', e.target.value)}
                className="flex-1 px-2 py-1 text-xs bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
              >
                <option value="">(use default)</option>
                {columns.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-text-muted w-20">Source</span>
              <select
                value={mapping.sourceColumn ?? ''}
                onChange={(e) => updateMapping('sourceColumn', e.target.value)}
                className="flex-1 px-2 py-1 text-xs bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
              >
                <option value="">(select)</option>
                {columns.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-text-muted w-20">Target</span>
              <select
                value={mapping.targetColumn ?? ''}
                onChange={(e) => updateMapping('targetColumn', e.target.value)}
                className="flex-1 px-2 py-1 text-xs bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
              >
                <option value="">(select)</option>
                {columns.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
          </>
        )}

        <div>
          <span className="text-xs text-text-muted">Property Columns</span>
          <div className="mt-1 flex flex-wrap gap-1">
            {columns
              .filter((col) =>
                col !== mapping.idColumn &&
                col !== mapping.labelsColumn &&
                col !== mapping.typeColumn &&
                col !== mapping.sourceColumn &&
                col !== mapping.targetColumn
              )
              .map((col) => (
                <label key={col} className="inline-flex items-center gap-1 text-xs">
                  <input
                    type="checkbox"
                    checked={mapping.propertyColumns.includes(col)}
                    onChange={(e) => {
                      const newCols = e.target.checked
                        ? [...mapping.propertyColumns, col]
                        : mapping.propertyColumns.filter((c) => c !== col)
                      updateMapping('propertyColumns', newCols)
                    }}
                    className="rounded border-border"
                  />
                  <span className="text-text-secondary">{col}</span>
                </label>
              ))}
          </div>
        </div>
      </div>

      {/* Preview */}
      {preview.rows.length > 0 && (
        <div>
          <label className="block text-xs text-text-muted mb-1">Preview (first 3 rows)</label>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-bg-tertiary">
                  {preview.headers.map((h, i) => (
                    <th key={i} className="px-2 py-1 text-left text-text-muted font-normal truncate max-w-[100px]">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.rows.slice(0, 3).map((row, i) => (
                  <tr key={i} className="border-t border-border">
                    {row.map((cell, j) => (
                      <td key={j} className="px-2 py-1 text-text-secondary truncate max-w-[100px]">
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

function ImportSection() {
  const [step, setStep] = useState<ImportStep>('select')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [fileContent, setFileContent] = useState<string>('')
  const [, setImportFormat] = useState<'json' | 'csv'>('json')
  const [progress, setProgress] = useState<ImportProgress | null>(null)
  const [result, setResult] = useState<ImportResult | null>(null)

  // CSV-specific state
  const [csvColumns, setCsvColumns] = useState<string[]>([])
  const [csvDataType, setCsvDataType] = useState<'nodes' | 'edges'>('nodes')
  const [csvDelimiter, setCsvDelimiter] = useState(',')
  const [csvMapping, setCsvMapping] = useState<CsvMapping>({
    idColumn: '',
    labelsColumn: undefined,
    typeColumn: undefined,
    sourceColumn: undefined,
    targetColumn: undefined,
    propertyColumns: [],
  })
  const [csvPreview, setCsvPreview] = useState<{ headers: string[]; rows: string[][] }>({
    headers: [],
    rows: [],
  })

  const runJsonImport = useCallback(async (content: string) => {
    setStep('progress')
    try {
      const importResult = await parseJsonImport(content, setProgress)
      setResult(importResult)
      setStep('result')

      if (importResult.errors.length === 0) {
        toast.success(`Imported ${importResult.stats.successfulNodes} nodes and ${importResult.stats.successfulEdges} edges`)
      } else {
        toast.warning(`Import completed with ${importResult.errors.length} warnings`)
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Import failed')
      setStep('select')
    }
  }, [])

  const handleFileSelect = useCallback(async (file: File) => {
    setSelectedFile(file)

    const isJson = file.name.toLowerCase().endsWith('.json')
    const isCsv = file.name.toLowerCase().endsWith('.csv')

    if (!isJson && !isCsv) {
      toast.error('Please select a JSON or CSV file')
      return
    }

    setImportFormat(isJson ? 'json' : 'csv')

    try {
      const content = await readFileAsText(file)
      setFileContent(content)

      if (isCsv) {
        const columns = detectCsvColumns(content, csvDelimiter)
        setCsvColumns(columns)
        setCsvMapping({
          idColumn: columns[0] ?? '',
          labelsColumn: undefined,
          typeColumn: undefined,
          sourceColumn: undefined,
          targetColumn: undefined,
          propertyColumns: columns.slice(1),
        })
        const preview = previewCsvData(content, csvDelimiter, 5)
        setCsvPreview(preview)
        setStep('configure')
      } else {
        // For JSON, go straight to import
        await runJsonImport(content)
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to read file')
    }
  }, [csvDelimiter, runJsonImport])

  const handleDelimiterChange = useCallback((delimiter: string) => {
    setCsvDelimiter(delimiter)
    if (fileContent) {
      const columns = detectCsvColumns(fileContent, delimiter)
      setCsvColumns(columns)
      const preview = previewCsvData(fileContent, delimiter, 5)
      setCsvPreview(preview)
      setCsvMapping({
        idColumn: columns[0] ?? '',
        labelsColumn: undefined,
        typeColumn: undefined,
        sourceColumn: undefined,
        targetColumn: undefined,
        propertyColumns: columns.slice(1),
      })
    }
  }, [fileContent])

  const runCsvImport = async () => {
    setStep('progress')
    try {
      const importResult = await parseCsvImport(
        fileContent,
        { delimiter: csvDelimiter, mapping: csvMapping, dataType: csvDataType },
        setProgress
      )
      setResult(importResult)
      setStep('result')

      if (importResult.errors.length === 0) {
        toast.success(`Imported ${importResult.stats.successfulNodes} nodes and ${importResult.stats.successfulEdges} edges`)
      } else {
        toast.warning(`Import completed with ${importResult.errors.length} warnings`)
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Import failed')
      setStep('configure')
    }
  }

  const handleReset = () => {
    setStep('select')
    setSelectedFile(null)
    setFileContent('')
    setProgress(null)
    setResult(null)
    setCsvColumns([])
    setCsvPreview({ headers: [], rows: [] })
  }

  return (
    <div className="space-y-4">
      {step === 'select' && (
        <FileDropZone
          onFileSelect={handleFileSelect}
          accept=".json,.csv"
        />
      )}

      {step === 'configure' && selectedFile && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileSpreadsheet size={16} className="text-text-muted" />
              <span className="text-sm text-text-primary truncate max-w-[150px]">
                {selectedFile.name}
              </span>
              <span className="text-xs text-text-muted">
                ({formatBytes(selectedFile.size)})
              </span>
            </div>
            <button
              onClick={handleReset}
              className="p-1 hover:bg-bg-tertiary rounded text-text-muted hover:text-text-primary"
            >
              <X size={14} />
            </button>
          </div>

          <CsvConfigPanel
            columns={csvColumns}
            dataType={csvDataType}
            onDataTypeChange={setCsvDataType}
            mapping={csvMapping}
            onMappingChange={setCsvMapping}
            preview={csvPreview}
            delimiter={csvDelimiter}
            onDelimiterChange={handleDelimiterChange}
          />

          <button
            onClick={runCsvImport}
            disabled={!csvMapping.idColumn || (csvDataType === 'edges' && (!csvMapping.sourceColumn || !csvMapping.targetColumn))}
            className="w-full py-2 px-4 bg-accent hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm rounded transition-colors"
          >
            Import {csvDataType}
          </button>
        </div>
      )}

      {step === 'progress' && progress && (
        <ImportProgressDisplay progress={progress} />
      )}

      {step === 'result' && result && (
        <ImportResultDisplay result={result} onReset={handleReset} />
      )}
    </div>
  )
}

function ExportSection() {
  const layout = useWorkspaceStore((s) => s.layout)
  const activePane = layout.panes[layout.activePaneId]
  const result = activePane?.result

  const [exportFormat, setExportFormat] = useState<'json' | 'csv'>('json')
  const [prettyPrint, setPrettyPrint] = useState(true)
  const [csvDelimiter, setCsvDelimiter] = useState(',')
  const [isExporting, setIsExporting] = useState(false)

  const hasData = useMemo(() => {
    if (!result) return false
    return (result.nodes?.length ?? 0) > 0 ||
           (result.edges?.length ?? 0) > 0 ||
           (result.rows?.length ?? 0) > 0
  }, [result])

  const dataStats = useMemo(() => {
    if (!result) return { nodes: 0, edges: 0, rows: 0 }
    return {
      nodes: result.nodes?.length ?? 0,
      edges: result.edges?.length ?? 0,
      rows: result.rows?.length ?? 0,
    }
  }, [result])

  const handleExport = async () => {
    if (!result) return

    setIsExporting(true)

    try {
      const options: ExportOptions = {
        format: exportFormat,
        prettyPrint,
        delimiter: csvDelimiter,
      }

      const content = exportFormat === 'json'
        ? exportQueryResultToJson(result, options)
        : exportQueryResultToCsv(result, options)

      const filename = generateExportFilename('export', exportFormat)
      const mimeType = exportFormat === 'json' ? 'application/json' : 'text/csv'

      downloadFile(content, filename, mimeType)
      toast.success(`Exported to ${filename}`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <div className="space-y-4">
      {!hasData ? (
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <Download size={32} className="text-text-muted/30 mb-3" />
          <p className="text-sm text-text-muted">No query results to export</p>
          <p className="text-xs text-text-muted/70 mt-1">
            Run a query first to export the results
          </p>
        </div>
      ) : (
        <>
          {/* Data Summary */}
          <div className="grid grid-cols-3 gap-2 text-center">
            {dataStats.nodes > 0 && (
              <div className="bg-bg-tertiary rounded-lg p-2">
                <p className="text-text-muted text-xs">Nodes</p>
                <p className="text-text-primary font-medium">{dataStats.nodes}</p>
              </div>
            )}
            {dataStats.edges > 0 && (
              <div className="bg-bg-tertiary rounded-lg p-2">
                <p className="text-text-muted text-xs">Edges</p>
                <p className="text-text-primary font-medium">{dataStats.edges}</p>
              </div>
            )}
            {dataStats.rows > 0 && (
              <div className="bg-bg-tertiary rounded-lg p-2">
                <p className="text-text-muted text-xs">Rows</p>
                <p className="text-text-primary font-medium">{dataStats.rows}</p>
              </div>
            )}
          </div>

          {/* Format Selection */}
          <div>
            <label className="block text-xs text-text-muted mb-1">Format</label>
            <div className="flex gap-2">
              <button
                onClick={() => setExportFormat('json')}
                className={`
                  flex items-center gap-2 flex-1 py-2 px-3 text-sm rounded transition-colors
                  ${exportFormat === 'json' ? 'bg-accent text-white' : 'bg-bg-tertiary text-text-secondary hover:bg-border'}
                `}
              >
                <FileJson size={16} />
                JSON
              </button>
              <button
                onClick={() => setExportFormat('csv')}
                className={`
                  flex items-center gap-2 flex-1 py-2 px-3 text-sm rounded transition-colors
                  ${exportFormat === 'csv' ? 'bg-accent text-white' : 'bg-bg-tertiary text-text-secondary hover:bg-border'}
                `}
              >
                <FileSpreadsheet size={16} />
                CSV
              </button>
            </div>
          </div>

          {/* Format Options */}
          {exportFormat === 'json' && (
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="prettyPrint"
                checked={prettyPrint}
                onChange={(e) => setPrettyPrint(e.target.checked)}
                className="rounded border-border"
              />
              <label htmlFor="prettyPrint" className="text-sm text-text-secondary">
                Pretty print (formatted)
              </label>
            </div>
          )}

          {exportFormat === 'csv' && (
            <div>
              <label className="block text-xs text-text-muted mb-1">Delimiter</label>
              <select
                value={csvDelimiter}
                onChange={(e) => setCsvDelimiter(e.target.value)}
                className="w-full px-2 py-1.5 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
              >
                <option value=",">Comma (,)</option>
                <option value=";">Semicolon (;)</option>
                <option value="\t">Tab</option>
                <option value="|">Pipe (|)</option>
              </select>
            </div>
          )}

          {/* Export Button */}
          <button
            onClick={handleExport}
            disabled={isExporting}
            className="w-full flex items-center justify-center gap-2 py-2 px-4 bg-accent hover:bg-accent-hover disabled:opacity-50 text-white text-sm rounded transition-colors"
          >
            {isExporting ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Exporting...
              </>
            ) : (
              <>
                <Download size={16} />
                Export {exportFormat.toUpperCase()}
              </>
            )}
          </button>
        </>
      )}
    </div>
  )
}

export function ImportExportPanel() {
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="text-lg font-semibold text-text-primary">Import / Export</h2>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <CollapsibleSection
          title="Import"
          icon={<Upload size={16} />}
          defaultOpen={true}
        >
          <div className="px-4 pb-2">
            <ImportSection />
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="Export"
          icon={<Download size={16} />}
          defaultOpen={true}
        >
          <div className="px-4 pb-2">
            <ExportSection />
          </div>
        </CollapsibleSection>
      </div>

      {/* Footer */}
      <div className="px-4 py-2 border-t border-border bg-bg-secondary">
        <p className="text-xs text-text-muted">
          Supports JSON and CSV formats
        </p>
      </div>
    </div>
  )
}

"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import {
  Database, FileText, BarChart3, AlertTriangle, CloudUpload,
  ArrowRight, Loader2, Trash2, RefreshCw, Pencil, Check, X,
} from "lucide-react"
import { toast } from "sonner"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const LS_SESSIONS = "modelforge_sessions_v2"

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

interface StoredDataset {
  file_id: string
  filename: string
  rows: number
  columns: string[]
  text_columns: string[]
  label_columns: string[]
  unique_labels: string[]
  class_distribution: Record<string, number>
  duplicate_count: number
  null_count: number
  sessionCount: number
  lastUsed: number
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function DatasetsClient() {
  const router = useRouter()
  const [datasets, setDatasets]         = useState<StoredDataset[]>([])
  const [checking, setChecking]         = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [renaming, setRenaming]         = useState<string | null>(null)
  const [renameValue, setRenameValue]   = useState("")

  function loadFromStorage() {
    try {
      const raw = localStorage.getItem(LS_SESSIONS)
      if (!raw) { setDatasets([]); return }

      const sessions: Array<{
        uploadResult?: StoredDataset | null
        createdAt?: number
      }> = JSON.parse(raw)

      const byId: Record<string, StoredDataset & { sessionCount: number; lastUsed: number }> = {}

      for (const s of sessions) {
        const ur = s.uploadResult
        if (!ur?.file_id) continue
        if (byId[ur.file_id]) {
          byId[ur.file_id].sessionCount++
          byId[ur.file_id].lastUsed = Math.max(byId[ur.file_id].lastUsed, s.createdAt ?? 0)
        } else {
          byId[ur.file_id] = { ...ur, sessionCount: 1, lastUsed: s.createdAt ?? 0 }
        }
      }

      setDatasets(Object.values(byId).sort((a, b) => b.lastUsed - a.lastUsed))
    } catch {
      setDatasets([])
    }
  }

  useEffect(() => { loadFromStorage() }, [])

  async function trainWithDataset(ds: StoredDataset) {
    setChecking(ds.file_id)
    try {
      const res = await fetch(`${API_URL}/health`)
      if (!res.ok) throw new Error("Backend unavailable")
      const uploadResult = {
        file_id:              ds.file_id,
        filename:             ds.filename,
        rows:                 ds.rows,
        columns:              ds.columns,
        text_columns:         ds.text_columns,
        label_columns:        ds.label_columns,
        unique_labels:        ds.unique_labels,
        class_distribution:   ds.class_distribution,
        text_length_stats:    {},
        text_length_histogram:[],
        data_warnings:        [],
        duplicate_count:      ds.duplicate_count,
        null_count:           ds.null_count,
        sample_rows:          [],
      }
      localStorage.setItem("modelforge_preselect_dataset", JSON.stringify(uploadResult))
      router.push("/train")
    } catch {
      toast.error("Backend is offline. Start the backend to use this dataset.")
    } finally {
      setChecking(null)
    }
  }

  async function handleDelete(file_id: string) {
    try {
      await fetch(`${API_URL}/datasets/${file_id}`, { method: "DELETE" })
    } catch { /* backend offline — still clean localStorage */ }

    try {
      const raw = localStorage.getItem(LS_SESSIONS)
      if (raw) {
        const sessions = JSON.parse(raw)
        const filtered = sessions.filter(
          (s: { uploadResult?: { file_id?: string } }) =>
            s.uploadResult?.file_id !== file_id
        )
        localStorage.setItem(LS_SESSIONS, JSON.stringify(filtered))
      }
    } catch {}

    setConfirmDelete(null)
    loadFromStorage()
    toast.success("Dataset removed.")
  }

  async function handleRename(file_id: string, newName: string) {
    const trimmed = newName.trim()
    if (!trimmed) { toast.error("Name cannot be empty."); return }

    try {
      await fetch(`${API_URL}/datasets/${file_id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: trimmed }),
      })
    } catch {}

    try {
      const raw = localStorage.getItem(LS_SESSIONS)
      if (raw) {
        const sessions = JSON.parse(raw)
        const updated = sessions.map((s: { uploadResult?: { file_id?: string; filename?: string } }) => {
          if (s.uploadResult?.file_id === file_id) {
            return { ...s, uploadResult: { ...s.uploadResult, filename: trimmed } }
          }
          return s
        })
        localStorage.setItem(LS_SESSIONS, JSON.stringify(updated))
      }
    } catch {}

    setRenaming(null)
    loadFromStorage()
    toast.success("Dataset renamed.")
  }

  function startRename(ds: StoredDataset) {
    setRenaming(ds.file_id)
    setRenameValue(ds.filename)
    setConfirmDelete(null)
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dataset Library</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Datasets from your training sessions — reuse them without re-uploading.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={loadFromStorage} className="gap-1.5">
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </Button>
          <Button size="sm" onClick={() => router.push("/train")} className="gap-1.5">
            <CloudUpload className="h-3.5 w-3.5" /> Upload New
          </Button>
        </div>
      </div>

      {/* Content */}
      {datasets.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 gap-4 text-center">
          <div className="h-14 w-14 rounded-2xl bg-secondary flex items-center justify-center">
            <Database className="h-7 w-7 text-muted-foreground" />
          </div>
          <div className="space-y-1">
            <p className="font-medium">No datasets yet</p>
            <p className="text-sm text-muted-foreground max-w-xs">
              Upload a dataset in the training workspace. It will appear here automatically.
            </p>
          </div>
          <Button onClick={() => router.push("/train")} className="gap-2">
            <CloudUpload className="h-4 w-4" /> Go to Training
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {datasets.map(ds => (
            <DatasetCard
              key={ds.file_id}
              ds={ds}
              checking={checking === ds.file_id}
              confirming={confirmDelete === ds.file_id}
              isRenaming={renaming === ds.file_id}
              renameValue={renaming === ds.file_id ? renameValue : ""}
              onTrain={() => trainWithDataset(ds)}
              onDelete={() => handleDelete(ds.file_id)}
              onConfirmDelete={() => setConfirmDelete(ds.file_id)}
              onCancelDelete={() => setConfirmDelete(null)}
              onStartRename={() => startRename(ds)}
              onRenameChange={setRenameValue}
              onRenameCommit={() => handleRename(ds.file_id, renameValue)}
              onRenameCancel={() => setRenaming(null)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Dataset Card
// ──────────────────────────────────────────────────────────────────────────────

function DatasetCard({
  ds, checking, confirming, isRenaming, renameValue,
  onTrain, onDelete, onConfirmDelete, onCancelDelete,
  onStartRename, onRenameChange, onRenameCommit, onRenameCancel,
}: {
  ds: StoredDataset
  checking: boolean
  confirming: boolean
  isRenaming: boolean
  renameValue: string
  onTrain: () => void
  onDelete: () => void
  onConfirmDelete: () => void
  onCancelDelete: () => void
  onStartRename: () => void
  onRenameChange: (v: string) => void
  onRenameCommit: () => void
  onRenameCancel: () => void
}) {
  const topLabels = Object.entries(ds.class_distribution)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)

  const totalRows = Object.values(ds.class_distribution).reduce((a, b) => a + b, 0) || ds.rows

  return (
    <Card className="group hover:border-primary/40 transition-colors">
      <CardContent className="p-4 space-y-3">
        {/* Header */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <div className="h-8 w-8 shrink-0 rounded-md bg-primary/10 flex items-center justify-center">
              <FileText className="h-4 w-4 text-primary" />
            </div>
            <div className="min-w-0 flex-1">
              {isRenaming ? (
                <div className="flex items-center gap-1">
                  <input
                    autoFocus
                    value={renameValue}
                    onChange={e => onRenameChange(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === "Enter") onRenameCommit()
                      if (e.key === "Escape") onRenameCancel()
                    }}
                    className="flex-1 min-w-0 px-1.5 py-0.5 text-sm font-semibold bg-background border border-primary rounded focus:outline-none"
                  />
                  <button onClick={onRenameCommit} className="text-emerald-400 hover:text-emerald-300">
                    <Check className="h-3.5 w-3.5" />
                  </button>
                  <button onClick={onRenameCancel} className="text-muted-foreground hover:text-foreground">
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ) : (
                <p className="font-semibold text-sm truncate">{ds.filename}</p>
              )}
              <p className="text-[10px] text-muted-foreground font-mono">{ds.file_id.slice(0, 8)}…</p>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1 shrink-0">
            <span className="text-[10px] px-2 py-0.5 rounded bg-secondary text-muted-foreground border border-border">
              {ds.sessionCount} session{ds.sessionCount !== 1 ? "s" : ""}
            </span>
            <button
              onClick={onStartRename}
              title="Rename"
              className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
            >
              <Pencil className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={onConfirmDelete}
              title="Delete"
              className="p-1 rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Delete confirmation strip */}
        {confirming && (
          <div className="flex items-center gap-2 p-2.5 rounded-lg bg-destructive/10 border border-destructive/20 text-xs">
            <AlertTriangle className="h-3.5 w-3.5 text-destructive shrink-0" />
            <span className="flex-1 text-destructive">
              Delete this dataset and {ds.sessionCount} session{ds.sessionCount !== 1 ? "s" : ""}?
            </span>
            <button onClick={onDelete} className="text-destructive font-medium hover:underline">Delete</button>
            <button onClick={onCancelDelete} className="text-muted-foreground hover:underline">Cancel</button>
          </div>
        )}

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-2">
          {([
            { label: "Rows",   value: ds.rows.toLocaleString(), icon: Database },
            { label: "Labels", value: String(ds.unique_labels.length), icon: BarChart3 },
            { label: "Dupes",  value: String(ds.duplicate_count), icon: AlertTriangle },
          ] as const).map(({ label, value, icon: Icon }) => (
            <div key={label} className="rounded-md bg-secondary/50 px-2 py-2 text-center">
              <p className="text-sm font-bold">{value}</p>
              <p className="text-[10px] text-muted-foreground">{label}</p>
            </div>
          ))}
        </div>

        {/* Column badges */}
        <div className="flex flex-wrap gap-1">
          {ds.text_columns.map(c => (
            <span key={c} className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
              {c} <span className="opacity-60">TEXT</span>
            </span>
          ))}
          {ds.label_columns.map(c => (
            <span key={c} className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
              {c} <span className="opacity-60">LABEL</span>
            </span>
          ))}
        </div>

        {/* Label distribution mini-bars */}
        {topLabels.length > 0 && (
          <div className="space-y-1">
            <p className="text-[10px] font-medium text-muted-foreground">Class distribution</p>
            {topLabels.map(([label, count]) => {
              const pct = Math.round((count / totalRows) * 100)
              return (
                <div key={label} className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground w-20 truncate">{label}</span>
                  <div className="flex-1 h-1.5 bg-secondary rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-primary/60" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="text-[10px] text-muted-foreground w-8 text-right">{pct}%</span>
                </div>
              )
            })}
            {ds.unique_labels.length > 5 && (
              <p className="text-[10px] text-muted-foreground">+{ds.unique_labels.length - 5} more labels</p>
            )}
          </div>
        )}

        {/* CTA */}
        <Button size="sm" className="w-full gap-1.5" onClick={onTrain} disabled={checking}>
          {checking
            ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
            : <ArrowRight className="h-3.5 w-3.5" />}
          {checking ? "Checking…" : "Train with this dataset"}
        </Button>
      </CardContent>
    </Card>
  )
}

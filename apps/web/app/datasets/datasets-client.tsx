"use client"

import { useState, useEffect, useCallback } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Database, FileText, BarChart3, AlertTriangle, CloudUpload,
  ArrowRight, Loader2, Trash2, RefreshCw,
} from "lucide-react"
import { toast } from "sonner"
import { cn } from "@/lib/utils"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const LS_SESSIONS = "modelforge_sessions_v2"

// ──────────────────────────────────────────────────────────────────────────────
// Types (mirrors UploadResult from train-client)
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
  // derived
  sessionCount: number
  lastUsed: number
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function DatasetsClient() {
  const router = useRouter()
  const [datasets, setDatasets] = useState<StoredDataset[]>([])
  const [checking, setChecking] = useState<string | null>(null)

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

      setDatasets(
        Object.values(byId).sort((a, b) => b.lastUsed - a.lastUsed)
      )
    } catch {
      setDatasets([])
    }
  }

  useEffect(() => { loadFromStorage() }, [])

  async function trainWithDataset(ds: StoredDataset) {
    // Verify file still exists on backend before navigating
    setChecking(ds.file_id)
    try {
      const res = await fetch(`${API_URL}/health`)
      if (!res.ok) throw new Error("Backend unavailable")
      // Backend is up — assume file is still registered (FILE_REGISTRY persists via .meta.json)
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
              onTrain={() => trainWithDataset(ds)}
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
  ds, checking, onTrain,
}: {
  ds: StoredDataset
  checking: boolean
  onTrain: () => void
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
          <div className="flex items-center gap-2 min-w-0">
            <div className="h-8 w-8 shrink-0 rounded-md bg-primary/10 flex items-center justify-center">
              <FileText className="h-4 w-4 text-primary" />
            </div>
            <div className="min-w-0">
              <p className="font-semibold text-sm truncate">{ds.filename}</p>
              <p className="text-[10px] text-muted-foreground font-mono">{ds.file_id.slice(0, 8)}…</p>
            </div>
          </div>
          <span className="text-[10px] shrink-0 px-2 py-0.5 rounded bg-secondary text-muted-foreground border border-border">
            {ds.sessionCount} session{ds.sessionCount !== 1 ? "s" : ""}
          </span>
        </div>

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
            {topLabels.map(([label, count], i) => {
              const pct = Math.round((count / totalRows) * 100)
              return (
                <div key={label} className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground w-20 truncate">{label}</span>
                  <div className="flex-1 h-1.5 bg-secondary rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary/60"
                      style={{ width: `${pct}%` }}
                    />
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
        <Button
          size="sm"
          className="w-full gap-1.5"
          onClick={onTrain}
          disabled={checking}
        >
          {checking
            ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
            : <ArrowRight className="h-3.5 w-3.5" />
          }
          {checking ? "Checking…" : "Train with this dataset"}
        </Button>
      </CardContent>
    </Card>
  )
}

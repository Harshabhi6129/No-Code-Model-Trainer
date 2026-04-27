"use client"

import { useState, useRef, useCallback } from "react"
import { useRouter } from "next/navigation"
import { AppShell } from "@/components/layout/app-shell"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import {
  Upload, Send, Loader2, CheckCircle2, XCircle,
  FileText, Database, BrainCircuit, Cpu, BarChart3, Rocket, CloudUpload
} from "lucide-react"
import { toast } from "sonner"
import { createClient } from "@/lib/supabase/client"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

interface AgentMessage { agent: string; success: boolean; message: string; output: Record<string, unknown> }
interface UploadResult {
  file_id: string; filename: string; rows: number
  columns: string[]; text_columns: string[]; label_columns: string[]; unique_labels: string[]
}

const STEPS = [
  { key: "upload", label: "Upload Dataset", icon: Upload },
  { key: "describe", label: "Describe Task",   icon: BrainCircuit },
  { key: "train",   label: "Train & Monitor",  icon: Cpu },
]

const AGENT_ICONS: Record<string, React.ElementType> = {
  Intent: BrainCircuit, Data: Database, Model: Cpu,
  Train: Cpu, Eval: BarChart3, Deploy: Rocket, System: XCircle,
}

const EXAMPLES = [
  "Classify customer support tickets by urgency: low, medium, high",
  "Detect toxic comments in user-generated content",
  "Extract product names and prices from e-commerce descriptions",
  "Fine-tune a model to answer FAQs about our software product",
]

export default function TrainPage() {
  const router = useRouter()
  const supabase = createClient()
  const fileRef = useRef<HTMLInputElement>(null)

  const [step, setStep] = useState(0)
  const [upload, setUpload] = useState<UploadResult | null>(null)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [message, setMessage] = useState("")
  const [messages, setMessages] = useState<AgentMessage[]>([])
  const [streaming, setStreaming] = useState(false)
  const [runId, setRunId] = useState<string | null>(null)

  const handleFile = useCallback(async (file: File) => {
    setUploading(true)
    const form = new FormData()
    form.append("file", file)
    try {
      const res = await fetch(`${API_URL}/upload`, { method: "POST", body: form })
      if (!res.ok) throw new Error(await res.text())
      const data: UploadResult = await res.json()
      setUpload(data)
      setStep(1)
      toast.success(`Loaded ${data.rows.toLocaleString()} rows from ${data.filename}`)
    } catch (err) {
      toast.error(`Upload failed: ${err}`)
    } finally {
      setUploading(false)
    }
  }, [])

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const completedAgents = messages.filter((m) => m.success).map((m) => m.agent)

  async function handleSend() {
    if (!message.trim()) return
    const intent = message.trim()
    setMessage("")
    setStep(2)
    setStreaming(true)

    // Create run record in Supabase
    const { data: { user } } = await supabase.auth.getUser()
    let newRunId: string | null = null
    if (user) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: run } = await (supabase as any).from("runs").insert({
        user_id: user.id,
        status: "running",
        dataset_filename: upload?.filename ?? null,
        dataset_rows: upload?.rows ?? null,
      }).select().single()
      newRunId = run?.id ?? null
      setRunId(newRunId)
    }

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: intent, file_id: upload?.file_id ?? null }),
      })
      if (!res.ok) throw new Error(await res.text())

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buf = ""
      const allMessages: AgentMessage[] = []

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split("\n")
        buf = lines.pop() ?? ""
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          const payload = line.slice(6).trim()
          if (payload === "[DONE]") break
          try {
            const msg: AgentMessage = JSON.parse(payload)
            allMessages.push(msg)
            setMessages([...allMessages])
          } catch {}
        }
      }

      const success = allMessages.every((m) => m.success)
      if (newRunId) {
        const lastMsg = allMessages[allMessages.length - 1]
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase as any).from("runs").update({
          status: success ? "completed" : "failed",
          task_type: (allMessages.find(m => m.agent === "Intent")?.output as Record<string,unknown>)?.task_type as string ?? null,
          model_id: (allMessages.find(m => m.agent === "Model")?.output as Record<string,unknown>)?.base_model as string ?? null,
          intent_spec: allMessages.find(m => m.agent === "Intent")?.output ?? {},
          model_recipe: allMessages.find(m => m.agent === "Model")?.output ?? {},
          completed_at: new Date().toISOString(),
          error_message: success ? null : (lastMsg?.message ?? null),
        }).eq("id", newRunId)
      }

      if (success) toast.success("Pipeline completed successfully!")
      else toast.error("Pipeline encountered an error — see details below")
    } catch (err) {
      toast.error(`Pipeline error: ${err}`)
      setMessages((prev) => [...prev, { agent: "System", success: false, message: String(err), output: {} }])
      if (newRunId) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase as any).from("runs").update({ status: "failed", error_message: String(err) }).eq("id", newRunId)
      }
    } finally {
      setStreaming(false)
    }
  }

  return (
    <AppShell>
      <div className="p-8 max-w-6xl mx-auto space-y-6">
        {/* Header + steps */}
        <div>
          <h1 className="text-2xl font-bold mb-1">New Training Run</h1>
          <p className="text-muted-foreground text-sm">Upload a dataset, describe your task, and watch agents train your model.</p>
        </div>

        <div className="flex items-center gap-0">
          {STEPS.map((s, i) => (
            <div key={s.key} className="flex items-center gap-0 flex-1">
              <div className={`flex items-center gap-2.5 px-4 py-2.5 rounded-lg transition-colors ${
                i === step ? "bg-primary/10 text-primary"
                : i < step ? "text-emerald-400"
                : "text-muted-foreground"
              }`}>
                {i < step
                  ? <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
                  : <s.icon className="h-4 w-4 shrink-0" />
                }
                <span className="text-sm font-medium">{s.label}</span>
              </div>
              {i < STEPS.length - 1 && (
                <div className={`flex-1 h-px mx-2 transition-colors ${i < step ? "bg-emerald-500/30" : "bg-border"}`} />
              )}
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column */}
          <div className="space-y-4">
            {/* Upload card */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <FileText className="h-4 w-4 text-primary" /> Dataset
                </CardTitle>
                <CardDescription>CSV, JSON, or JSONL file</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <input ref={fileRef} type="file" accept=".csv,.json,.jsonl" className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />

                {!upload ? (
                  <div
                    onDrop={onDrop}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                    onDragLeave={() => setDragOver(false)}
                    onClick={() => fileRef.current?.click()}
                    className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center gap-3 cursor-pointer transition-colors ${
                      dragOver ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-secondary/50"
                    }`}
                  >
                    {uploading
                      ? <Loader2 className="h-8 w-8 text-primary animate-spin" />
                      : <CloudUpload className="h-8 w-8 text-muted-foreground" />
                    }
                    <div className="text-center">
                      <p className="text-sm font-medium">{uploading ? "Uploading…" : "Drop file here"}</p>
                      <p className="text-xs text-muted-foreground mt-0.5">or click to browse</p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="flex items-start gap-3 p-3 rounded-lg bg-emerald-500/5 border border-emerald-500/20">
                      <CheckCircle2 className="h-4 w-4 text-emerald-400 mt-0.5 shrink-0" />
                      <div className="min-w-0">
                        <p className="text-sm font-medium truncate">{upload.filename}</p>
                        <p className="text-xs text-muted-foreground">{upload.rows.toLocaleString()} rows · {upload.columns.length} columns</p>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {upload.columns.slice(0, 8).map((c) => (
                        <Badge key={c} variant="secondary" className="text-[10px] px-1.5 h-5">{c}</Badge>
                      ))}
                    </div>
                    {upload.unique_labels.length > 0 && (
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">Labels detected:</p>
                        <div className="flex flex-wrap gap-1">
                          {upload.unique_labels.slice(0, 6).map((l) => (
                            <Badge key={l} variant="outline" className="text-[10px] px-1.5 h-5">{l}</Badge>
                          ))}
                          {upload.unique_labels.length > 6 && (
                            <Badge variant="outline" className="text-[10px] px-1.5 h-5">+{upload.unique_labels.length - 6}</Badge>
                          )}
                        </div>
                      </div>
                    )}
                    <Button variant="outline" size="sm" className="w-full text-xs"
                      onClick={() => { setUpload(null); setStep(0) }}>
                      Replace file
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Examples */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Try an example</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1">
                {EXAMPLES.map((ex) => (
                  <button key={ex}
                    className="w-full text-left text-xs text-muted-foreground hover:text-foreground hover:bg-secondary p-2 rounded-lg transition-colors leading-relaxed"
                    onClick={() => { setMessage(ex); if (step === 0 && !upload) toast.info("Upload a dataset first, or send to test intent detection") }}>
                    "{ex}"
                  </button>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Right column — pipeline */}
          <div className="lg:col-span-2 flex flex-col gap-4">
            <Card className="flex-1 flex flex-col min-h-[480px]">
              <CardHeader className="pb-3 shrink-0">
                <CardTitle className="text-sm flex items-center gap-2">
                  <BrainCircuit className="h-4 w-4 text-primary" /> Agent Pipeline
                </CardTitle>
                <CardDescription>Watch the agents work through your request in real time</CardDescription>
              </CardHeader>

              {streaming && (
                <div className="px-6 shrink-0">
                  <div className="flex items-center justify-between text-xs text-muted-foreground mb-1.5">
                    <span>Pipeline running…</span>
                    <span>{completedAgents.length} / 3 agents done</span>
                  </div>
                  <Progress value={(completedAgents.length / 3) * 100} className="h-1.5" />
                </div>
              )}

              <Separator className="my-3" />

              <CardContent className="flex-1 overflow-y-auto space-y-4 max-h-[360px]">
                {messages.length === 0 && !streaming && (
                  <div className="flex flex-col items-center justify-center h-48 gap-3 text-center">
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-secondary">
                      <BrainCircuit className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <div>
                      <p className="text-sm font-medium">Waiting for your task description</p>
                      <p className="text-xs text-muted-foreground mt-1">Type below or pick an example →</p>
                    </div>
                  </div>
                )}

                {messages.map((msg, i) => {
                  const AgentIcon = AGENT_ICONS[msg.agent] ?? BrainCircuit
                  return (
                    <div key={i} className={`rounded-xl p-4 border ${msg.success ? "bg-secondary/50 border-border" : "bg-destructive/5 border-destructive/30"}`}>
                      <div className="flex items-center gap-2 mb-2">
                        <div className={`flex h-6 w-6 items-center justify-center rounded-md ${msg.success ? "bg-primary/10" : "bg-destructive/10"}`}>
                          <AgentIcon className={`h-3.5 w-3.5 ${msg.success ? "text-primary" : "text-destructive"}`} />
                        </div>
                        <Badge variant={msg.success ? "outline" : "destructive"} className="text-[10px] h-4 px-1.5">
                          {msg.agent} Agent
                        </Badge>
                        {msg.success
                          ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 ml-auto" />
                          : <XCircle className="h-3.5 w-3.5 text-destructive ml-auto" />
                        }
                      </div>
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.message}</p>
                    </div>
                  )
                })}

                {streaming && (
                  <div className="flex items-center gap-3 text-sm text-muted-foreground animate-pulse">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    Agent is processing…
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Input */}
            <div className="flex gap-2">
              <input
                className="flex-1 h-10 rounded-lg border border-input bg-card px-4 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary transition-colors"
                placeholder={step === 0 ? "Upload a dataset to get started, or type to test intent…" : "Describe your ML task in plain English…"}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
                disabled={streaming}
              />
              <Button onClick={handleSend} disabled={streaming || !message.trim()} className="h-10 px-4">
                {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>

            {runId && !streaming && messages.length > 0 && (
              <Button variant="outline" size="sm" className="w-full" onClick={() => router.push(`/runs/${runId}`)}>
                View full run details →
              </Button>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  )
}

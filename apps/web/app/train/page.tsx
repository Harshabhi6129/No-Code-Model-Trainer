"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Upload, Send, Loader2, CheckCircle2, XCircle, BrainCircuit } from "lucide-react"
import Link from "next/link"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

interface AgentMessage {
  agent: string
  success: boolean
  message: string
  output: Record<string, unknown>
}

interface UploadResult {
  file_id: string
  file_path: string
  filename: string
  rows: number
  columns: string[]
  text_columns: string[]
  label_columns: string[]
  unique_labels: string[]
}

export default function TrainPage() {
  const [upload, setUpload] = useState<UploadResult | null>(null)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState("")
  const [messages, setMessages] = useState<AgentMessage[]>([])
  const [streaming, setStreaming] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    const form = new FormData()
    form.append("file", file)
    try {
      const res = await fetch(`${API_URL}/upload`, { method: "POST", body: form })
      if (!res.ok) throw new Error(await res.text())
      setUpload(await res.json())
    } catch (err) {
      alert(`Upload failed: ${err}`)
    } finally {
      setUploading(false)
    }
  }

  async function handleSend() {
    if (!message.trim()) return
    const userMsg = message.trim()
    setMessage("")
    setStreaming(true)

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg, dataset_path: upload?.file_path ?? null }),
      })
      if (!res.ok) throw new Error(await res.text())
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buf = ""
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
            setMessages((prev) => [...prev, msg])
          } catch {}
        }
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { agent: "System", success: false, message: String(err), output: {} },
      ])
    } finally {
      setStreaming(false)
    }
  }

  return (
    <main className="flex flex-col min-h-screen">
      <nav className="border-b px-6 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-semibold text-lg">
          <BrainCircuit className="h-5 w-5" />
          ModelForge
        </Link>
        <Button variant="ghost" asChild>
          <Link href="/dashboard">Dashboard</Link>
        </Button>
      </nav>

      <div className="flex flex-1 gap-6 p-6 max-w-6xl mx-auto w-full">
        {/* Left panel — dataset */}
        <div className="w-72 shrink-0 flex flex-col gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Dataset</CardTitle>
              <CardDescription>Upload a CSV or JSON file</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <input
                ref={fileRef}
                type="file"
                accept=".csv,.json,.jsonl"
                className="hidden"
                onChange={handleUpload}
              />
              <Button
                variant="outline"
                className="w-full"
                onClick={() => fileRef.current?.click()}
                disabled={uploading}
              >
                {uploading ? (
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <Upload className="h-4 w-4 mr-2" />
                )}
                {uploading ? "Uploading…" : "Choose File"}
              </Button>

              {upload && (
                <div className="text-sm space-y-1">
                  <p className="font-medium truncate">{upload.filename}</p>
                  <p className="text-muted-foreground">{upload.rows.toLocaleString()} rows</p>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {upload.columns.slice(0, 6).map((c) => (
                      <Badge key={c} variant="secondary" className="text-xs">{c}</Badge>
                    ))}
                  </div>
                  {upload.unique_labels.length > 0 && (
                    <p className="text-muted-foreground text-xs mt-1">
                      Labels: {upload.unique_labels.slice(0, 5).join(", ")}
                      {upload.unique_labels.length > 5 && "…"}
                    </p>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Examples</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-2">
              {[
                "Classify customer support tickets by urgency",
                "Fine-tune a model to detect toxic comments",
                "Train a NER model to extract product names",
              ].map((ex) => (
                <button
                  key={ex}
                  className="text-left text-xs text-muted-foreground hover:text-foreground p-2 rounded hover:bg-secondary transition-colors"
                  onClick={() => setMessage(ex)}
                >
                  {ex}
                </button>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Right panel — chat */}
        <div className="flex-1 flex flex-col gap-4">
          <Card className="flex-1 flex flex-col">
            <CardHeader>
              <CardTitle className="text-base">Agent Pipeline</CardTitle>
              <CardDescription>Describe your ML task and watch the agents work</CardDescription>
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto flex flex-col gap-3 min-h-0 max-h-[500px]">
              {messages.length === 0 && (
                <p className="text-sm text-muted-foreground text-center mt-12">
                  Upload a dataset and describe your task to start.
                </p>
              )}
              {messages.map((msg, i) => (
                <div key={i} className="flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    {msg.success ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />
                    ) : (
                      <XCircle className="h-4 w-4 text-destructive shrink-0" />
                    )}
                    <Badge variant={msg.success ? "default" : "destructive"} className="text-xs">
                      {msg.agent} Agent
                    </Badge>
                  </div>
                  <p className="text-sm pl-6 whitespace-pre-wrap">{msg.message}</p>
                </div>
              ))}
              {streaming && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Agents are working…
                </div>
              )}
            </CardContent>
          </Card>

          <div className="flex gap-2">
            <input
              className="flex-1 border rounded-md px-3 py-2 text-sm bg-background focus:outline-none focus:ring-1 focus:ring-ring"
              placeholder="Describe your ML task in plain English…"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
              disabled={streaming}
            />
            <Button onClick={handleSend} disabled={streaming || !message.trim()}>
              {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>
    </main>
  )
}

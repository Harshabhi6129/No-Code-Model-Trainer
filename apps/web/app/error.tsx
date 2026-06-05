"use client"

import { useEffect } from "react"
import { Button } from "@/components/ui/button"
import { BrainCircuit, RefreshCw, Mail } from "lucide-react"

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    // Log to console in dev; production would send to Sentry
    console.error(error)
  }, [error])

  return (
    <div
      className="min-h-screen flex items-center justify-center grain relative overflow-hidden"
      style={{ background: "rgb(6,10,16)" }}
    >
      <div className="aurora-bg" aria-hidden>
        <div className="aurora-orb aurora-1" />
        <div className="aurora-orb aurora-3" />
      </div>

      <div className="relative z-10 flex flex-col items-center gap-8 text-center px-8 max-w-md">
        <div
          className="flex h-14 w-14 items-center justify-center rounded-2xl"
          style={{
            background: "rgba(239,68,68,0.1)",
            border: "1px solid rgba(239,68,68,0.3)",
            boxShadow: "0 0 30px -4px rgba(239,68,68,0.3)",
          }}
        >
          <BrainCircuit className="h-7 w-7 text-rose-400" />
        </div>

        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-foreground">Something went wrong</h1>
          <p className="text-sm text-muted-foreground max-w-xs leading-relaxed">
            An unexpected error occurred. Your training sessions are saved — nothing was lost.
          </p>
          {error.digest && (
            <p className="text-[10px] font-mono text-muted-foreground/40">Error ID: {error.digest}</p>
          )}
        </div>

        <div className="flex gap-3">
          <Button onClick={reset} className="gap-2">
            <RefreshCw className="h-4 w-4" />
            Try again
          </Button>
          <Button variant="outline" asChild>
            <a href="mailto:harsha6129abhi@gmail.com?subject=ModelForge error report" className="gap-2">
              <Mail className="h-4 w-4" />
              Report
            </a>
          </Button>
        </div>
      </div>
    </div>
  )
}

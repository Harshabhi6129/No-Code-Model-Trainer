import Link from "next/link"
import { Button } from "@/components/ui/button"
import { BrainCircuit, Home } from "lucide-react"

export default function NotFound() {
  return (
    <div
      className="min-h-screen flex items-center justify-center grain relative overflow-hidden"
      style={{ background: "rgb(6,10,16)" }}
    >
      {/* Aurora orbs */}
      <div className="aurora-bg" aria-hidden>
        <div className="aurora-orb aurora-1" />
        <div className="aurora-orb aurora-2" />
        <div className="aurora-orb aurora-3" />
      </div>

      <div className="relative z-10 flex flex-col items-center gap-8 text-center px-8 max-w-md">
        {/* Logo */}
        <div
          className="flex h-14 w-14 items-center justify-center rounded-2xl animate-glow-breathe"
          style={{
            background: "linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15))",
            border: "1px solid rgba(99,102,241,0.35)",
            boxShadow: "0 0 30px -4px rgba(99,102,241,0.5)",
          }}
        >
          <BrainCircuit className="h-7 w-7" style={{ color: "#818CF8" }} />
        </div>

        {/* 404 */}
        <div className="space-y-2">
          <p className="text-8xl font-black font-mono text-shimmer leading-none">404</p>
          <h1 className="text-2xl font-bold text-foreground">Page not found</h1>
          <p className="text-sm text-muted-foreground max-w-xs leading-relaxed">
            The page you&apos;re looking for doesn&apos;t exist or has been moved.
          </p>
        </div>

        <Button asChild size="lg" className="gap-2">
          <Link href="/dashboard">
            <Home className="h-4 w-4" />
            Go to Dashboard
          </Link>
        </Button>
      </div>
    </div>
  )
}

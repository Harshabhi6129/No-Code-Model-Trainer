import { cn } from "@/lib/utils"
import type { LucideIcon } from "lucide-react"

interface PageHeaderProps {
  icon?: LucideIcon
  iconColor?: string
  title: string
  description?: string
  actions?: React.ReactNode
  className?: string
}

export function PageHeader({
  icon: Icon,
  iconColor = "text-muted-foreground",
  title,
  description,
  actions,
  className,
}: PageHeaderProps) {
  return (
    <div
      className={cn(
        "relative flex items-center justify-between px-8 py-5 overflow-hidden",
        className
      )}
      style={{
        background: "rgba(6,10,16,0.55)",
        backdropFilter: "blur(24px) saturate(1.8)",
        WebkitBackdropFilter: "blur(24px) saturate(1.8)",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
        boxShadow: "0 1px 0 rgba(99,102,241,0.08)",
      }}
    >
      {/* Gradient accent line at top */}
      <div
        className="absolute top-0 left-0 right-0 h-px pointer-events-none"
        style={{ background: "linear-gradient(90deg, transparent 0%, rgba(99,102,241,0.4) 30%, rgba(139,92,246,0.4) 70%, transparent 100%)" }}
      />

      {/* Ambient glow behind icon */}
      {Icon && (
        <div
          className="absolute left-8 top-1/2 -translate-y-1/2 h-16 w-16 rounded-full blur-2xl pointer-events-none opacity-30"
          style={{ background: "rgba(99,102,241,0.5)" }}
        />
      )}

      <div className="flex items-start gap-3 relative">
        {Icon && (
          <div
            className="flex h-9 w-9 items-center justify-center rounded-lg mt-0.5 shrink-0"
            style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
              boxShadow: "inset 0 1px 0 rgba(255,255,255,0.08)",
            }}
          >
            <Icon className={cn("h-4 w-4", iconColor)} />
          </div>
        )}
        <div>
          <h1 className="text-lg font-bold tracking-tight text-foreground">{title}</h1>
          {description && (
            <p className="text-sm text-muted-foreground mt-0.5">{description}</p>
          )}
        </div>
      </div>

      {actions && (
        <div className="flex items-center gap-2 shrink-0 relative">
          {actions}
        </div>
      )}
    </div>
  )
}

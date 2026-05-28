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
        "flex items-center justify-between px-8 py-5 border-b border-border/60",
        className
      )}
      style={{
        background: "linear-gradient(180deg, hsl(var(--surface) / 0.4) 0%, transparent 100%)",
      }}
    >
      <div className="flex items-start gap-3">
        {Icon && (
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-surface-elevated border border-border mt-0.5 shrink-0">
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
        <div className="flex items-center gap-2 shrink-0">
          {actions}
        </div>
      )}
    </div>
  )
}

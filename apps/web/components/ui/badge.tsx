import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11px] font-semibold tracking-wide font-mono transition-all",
  {
    variants: {
      variant: {
        default:     "bg-primary/15 text-indigo-300 border border-indigo-500/20",
        secondary:   "bg-surface-high text-muted-foreground border border-border",
        destructive: "bg-rose-500/12 text-rose-400 border border-rose-500/20",
        outline:     "border border-border text-muted-foreground",
        success:     "bg-emerald-500/12 text-emerald-400 border border-emerald-500/20",
        warning:     "bg-amber-500/12 text-amber-400 border border-amber-500/20",
        running:     "bg-cyan-500/12 text-cyan-400 border border-cyan-500/20",
        violet:      "bg-violet-500/12 text-violet-400 border border-violet-500/20",
      },
    },
    defaultVariants: { variant: "default" },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {
  dot?: boolean
  pulse?: boolean
}

function Badge({ className, variant, dot, pulse, children, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props}>
      {dot && (
        <span className={cn("relative inline-flex h-1.5 w-1.5 rounded-full shrink-0", pulse && "ping-dot")}>
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
        </span>
      )}
      {children}
    </div>
  )
}

export { Badge, badgeVariants }

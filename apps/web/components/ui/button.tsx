import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-40 active:scale-[0.97]",
  {
    variants: {
      variant: {
        default:
          "relative overflow-hidden bg-primary text-primary-foreground hover:brightness-115 shadow-[0_0_24px_-4px_rgba(99,102,241,0.55)] hover:shadow-[0_0_36px_-2px_rgba(99,102,241,0.75)] btn-shimmer",
        destructive:
          "bg-destructive text-destructive-foreground hover:brightness-110 shadow-[0_0_18px_-4px_rgba(239,68,68,0.45)]",
        outline:
          "border bg-transparent text-foreground hover:bg-white/5 hover:border-white/15 hover:text-foreground",
        secondary:
          "bg-surface-elevated text-foreground border border-border hover:bg-surface-high hover:border-border-bright",
        ghost:
          "text-muted-foreground hover:bg-white/5 hover:text-foreground border border-transparent hover:border-white/8",
        link:
          "text-primary underline-offset-4 hover:underline p-0 h-auto",
        glow:
          "relative overflow-hidden bg-gradient-to-r from-indigo-600 to-violet-600 text-white hover:from-indigo-500 hover:to-violet-500 shadow-[0_0_32px_-4px_rgba(99,102,241,0.65)] hover:shadow-[0_0_44px_-2px_rgba(139,92,246,0.8)] btn-shimmer",
      },
      size: {
        default:  "h-9 px-4 py-2",
        sm:       "h-8 rounded-md px-3 text-xs",
        lg:       "h-10 px-6",
        xl:       "h-11 px-8 text-base",
        icon:     "h-9 w-9",
        "icon-sm":"h-7 w-7 rounded-md",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }

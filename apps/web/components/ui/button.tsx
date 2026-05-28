import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-40 active:scale-[0.98]",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground hover:brightness-110 shadow-[0_0_20px_-4px_hsl(var(--indigo)/0.5)] hover:shadow-[0_0_28px_-2px_hsl(var(--indigo)/0.65)]",
        destructive:
          "bg-rose text-destructive-foreground hover:brightness-110 shadow-[0_0_16px_-4px_hsl(var(--rose)/0.4)]",
        outline:
          "border border-border-bright bg-surface-elevated text-foreground hover:bg-surface-high hover:border-border-bright/80 hover:text-foreground",
        secondary:
          "bg-surface-elevated text-foreground border border-border hover:bg-surface-high hover:border-border-bright",
        ghost:
          "text-muted-foreground hover:bg-surface-elevated hover:text-foreground border border-transparent hover:border-border/50",
        link:
          "text-primary underline-offset-4 hover:underline p-0 h-auto",
        glow:
          "bg-gradient-to-r from-indigo-600 to-violet-600 text-white hover:from-indigo-500 hover:to-violet-500 shadow-[0_0_28px_-4px_hsl(var(--indigo)/0.6)] hover:shadow-[0_0_36px_-2px_hsl(var(--violet)/0.7)]",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm:      "h-8 rounded-md px-3 text-xs",
        lg:      "h-10 px-6",
        xl:      "h-11 px-8 text-base",
        icon:    "h-9 w-9",
        "icon-sm": "h-7 w-7 rounded-md",
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

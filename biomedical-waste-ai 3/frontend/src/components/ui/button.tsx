import * as React from "react";
import { cn } from "@/lib/utils";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost" | "outline" | "danger" | "success";
  size?: "xs" | "sm" | "md" | "lg";
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { className, variant = "primary", size = "md", loading, children, disabled, ...props },
    ref
  ) => {
    const base =
      "relative inline-flex items-center justify-center gap-2 rounded-full font-medium " +
      "transition-all duration-200 ease-spring focus-visible:outline-none focus-visible:ring-2 " +
      "focus-visible:ring-primary/60 disabled:opacity-50 disabled:cursor-not-allowed " +
      "disabled:pointer-events-none select-none";

    const variants: Record<string, string> = {
      primary:
        "bg-gradient-to-r from-violet-600 via-purple-500 to-fuchsia-500 text-white " +
        "shadow-btn-primary hover:shadow-glow-primary hover:scale-[1.03] active:scale-[0.98]",
      ghost:
        "text-foreground/70 hover:text-foreground hover:bg-white/5 active:bg-white/10",
      outline:
        "border border-white/10 text-foreground/80 hover:text-foreground " +
        "hover:bg-white/5 hover:border-white/20 active:bg-white/10",
      danger:
        "bg-red-500/10 text-red-300 border border-red-500/30 " +
        "hover:bg-red-500/20 hover:border-red-500/50 active:bg-red-500/30",
      success:
        "bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 " +
        "hover:bg-emerald-500/20 hover:border-emerald-500/50",
    };

    const sizes: Record<string, string> = {
      xs: "h-7  px-3   text-xs",
      sm: "h-9  px-4   text-sm",
      md: "h-11 px-6   text-sm",
      lg: "h-12 px-8   text-base",
    };

    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={cn(base, variants[variant], sizes[size], className)}
        {...props}
      >
        {loading && (
          <svg
            className="h-4 w-4 animate-spin shrink-0"
            viewBox="0 0 24 24"
            fill="none"
          >
            <circle
              className="opacity-25"
              cx="12" cy="12" r="10"
              stroke="currentColor" strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.37 0 0 5.37 0 12h4z"
            />
          </svg>
        )}
        {children}
      </button>
    );
  }
);
Button.displayName = "Button";

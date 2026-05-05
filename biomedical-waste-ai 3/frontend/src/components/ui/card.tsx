import * as React from "react";
import { cn } from "@/lib/utils";

/* ── Card ─────────────────────────────────────────────────────────────── */
export const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & { hoverable?: boolean }
>(({ className, hoverable, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "glass rounded-2xl shadow-card",
      hoverable && "transition-all duration-200 hover:bg-white/[0.06] hover:-translate-y-0.5",
      className
    )}
    {...props}
  />
));
Card.displayName = "Card";

/* ── CardHeader ───────────────────────────────────────────────────────── */
export const CardHeader = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("flex items-center justify-between mb-4", className)} {...props} />
);

/* ── CardTitle ────────────────────────────────────────────────────────── */
export const CardTitle = ({
  className,
  icon,
  children,
}: React.HTMLAttributes<HTMLParagraphElement> & { icon?: React.ReactNode }) => (
  <div className={cn("flex items-center gap-2", className)}>
    {icon && (
      <span className="h-8 w-8 rounded-lg bg-white/5 flex items-center justify-center text-primary">
        {icon}
      </span>
    )}
    <p className="text-xs font-semibold uppercase tracking-widest text-foreground/50">
      {children}
    </p>
  </div>
);

/* ── CardValue ────────────────────────────────────────────────────────── */
export const CardValue = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) => (
  <p
    className={cn("text-3xl font-semibold text-foreground tabular-nums mt-1", className)}
    {...props}
  />
);

/* ── Badge ────────────────────────────────────────────────────────────── */
export const Badge = ({
  children,
  variant = "default",
  className,
}: {
  children: React.ReactNode;
  variant?: "default" | "hazard" | "warning" | "success" | "muted";
  className?: string;
}) => {
  const variants: Record<string, string> = {
    default: "bg-primary/10 text-primary border-primary/20",
    hazard:  "bg-red-500/10 text-red-300 border-red-500/30 animate-alert",
    warning: "bg-amber-500/10 text-amber-300 border-amber-500/30",
    success: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
    muted:   "bg-white/5 text-foreground/60 border-white/10",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider",
        variants[variant],
        className
      )}
    >
      {children}
    </span>
  );
};

/* ── Skeleton ─────────────────────────────────────────────────────────── */
export const Skeleton = ({ className }: { className?: string }) => (
  <div className={cn("shimmer rounded-xl bg-white/5", className)} />
);

/* ── Divider ──────────────────────────────────────────────────────────── */
export const Divider = ({ className }: { className?: string }) => (
  <div className={cn("h-px bg-white/[0.07] my-4", className)} />
);

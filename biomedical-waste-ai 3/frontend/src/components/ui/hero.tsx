import { type ReactNode } from "react";
import { Biohazard, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

interface HeroProps {
  children?: ReactNode;
  className?: string;
}

export function Hero({ children, className }: HeroProps) {
  return (
    <section
      className={cn(
        "relative min-h-screen w-full overflow-hidden bg-[#0c0414] flex items-center justify-center",
        className
      )}
    >
      {/* Layered ambient background */}
      <div className="absolute inset-0 bg-hero-glow pointer-events-none" />

      {/* Dot grid */}
      <div
        className="absolute inset-0 opacity-[0.06] pointer-events-none"
        style={{
          backgroundImage:
            "radial-gradient(circle, rgba(255,255,255,0.8) 1px, transparent 1px)",
          backgroundSize: "36px 36px",
        }}
      />

      {/* Animated orbs */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute top-[-12%] left-[35%] h-[700px] w-[700px] rounded-full bg-violet-600/20 blur-[120px] animate-orb" />
        <div className="absolute bottom-[-10%] left-[-10%] h-[500px] w-[500px] rounded-full bg-blue-600/15 blur-[100px] animate-orb-slow" style={{ animationDelay: "-4s" }} />
        <div className="absolute top-[20%] right-[-8%] h-[400px] w-[400px] rounded-full bg-fuchsia-600/12 blur-[90px] animate-orb" style={{ animationDelay: "-2s" }} />
      </div>

      {/* Content */}
      <div className="relative z-10 mx-auto w-full max-w-4xl px-6 py-28 flex flex-col items-center text-center">
        {/* Eyebrow chip */}
        <div className="inline-flex items-center gap-2.5 rounded-full border border-white/10 bg-white/[0.06] px-4 py-1.5 text-xs text-foreground/60 backdrop-blur-md animate-fade-in mb-8">
          <Sparkles className="h-3.5 w-3.5 text-violet-400" />
          <span>WHO-aligned  ·  YOLOv8 Detection  ·  EfficientNet-B0 Classification</span>
        </div>

        {/* Icon + Title */}
        <div className="flex items-center justify-center gap-4 mb-6 animate-fade-up" style={{ animationDelay: "0.1s" }}>
          <div className="relative">
            <div className="absolute inset-0 rounded-2xl bg-gradient-primary blur-xl opacity-50" />
            <div className="relative h-16 w-16 rounded-2xl bg-gradient-to-br from-violet-600 to-fuchsia-600 flex items-center justify-center shadow-glow-primary">
              <Biohazard className="h-8 w-8 text-white" />
            </div>
          </div>
        </div>

        <h1
          className="text-5xl md:text-6xl lg:text-7xl font-semibold tracking-tight leading-[1.04] animate-fade-up"
          style={{ animationDelay: "0.15s" }}
        >
          <span className="gradient-text">Biomedical Waste</span>
          <br />
          <span className="text-foreground/90">AI System</span>
        </h1>

        <p
          className="mt-6 text-base md:text-lg text-foreground/50 max-w-2xl leading-relaxed animate-fade-up"
          style={{ animationDelay: "0.25s" }}
        >
          Real-time detection via YOLOv8 → crop → EfficientNet-B0 classification
          with zero-LLM, rule-based WHO-compliant bin segregation and hazard alerts.
        </p>

        {/* Stats row */}
        <div
          className="mt-8 flex items-center gap-6 text-sm text-foreground/40 animate-fade-up"
          style={{ animationDelay: "0.3s" }}
        >
          {[
            ["6", "WHO categories"],
            ["5", "Bin types"],
            ["0.80", "Conf. threshold"],
          ].map(([val, label]) => (
            <div key={label} className="flex items-baseline gap-1.5">
              <span className="text-lg font-semibold text-foreground/70">{val}</span>
              <span>{label}</span>
            </div>
          ))}
        </div>

        {/* Slot for page-specific input */}
        <div className="mt-12 w-full animate-fade-up" style={{ animationDelay: "0.4s" }}>
          {children}
        </div>
      </div>
    </section>
  );
}

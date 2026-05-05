import { confColor, fmt } from "@/lib/utils";

interface Props {
  value: number;    // 0–1
  size?: number;
  label?: string;
}

export function ConfidenceRing({ value, size = 88, label }: Props) {
  const r = (size - 10) / 2;
  const circ = 2 * Math.PI * r;
  const dash = circ * value;
  const color = confColor(value);

  return (
    <div className="flex flex-col items-center gap-2">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{ transform: "rotate(-90deg)" }}
      >
        {/* Track */}
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={8}
        />
        {/* Fill */}
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke={color}
          strokeWidth={8}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circ - dash}`}
          style={{ transition: "stroke-dasharray 0.9s cubic-bezier(0.4,0,0.2,1)" }}
        />
        {/* Glow */}
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke={color}
          strokeWidth={4}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circ - dash}`}
          opacity={0.3}
          filter="blur(4px)"
        />
      </svg>
      {/* Number overlay — we rotate back */}
      <div
        className="absolute flex flex-col items-center justify-center"
        style={{ width: size, height: size, marginTop: -(size + 8) }}
      >
        <span className="text-xl font-semibold tabular-nums" style={{ color }}>
          {fmt(value)}%
        </span>
        {label && <span className="text-[9px] text-foreground/40 mt-0.5">{label}</span>}
      </div>
    </div>
  );
}

/* Compact inline bar version */
export function ConfBar({ value, label }: { value: number; label?: string }) {
  const color = confColor(value);
  return (
    <div className="w-full">
      {label && (
        <div className="flex items-center justify-between mb-1.5 text-xs text-foreground/50">
          <span>{label}</span>
          <span style={{ color }} className="font-medium">{fmt(value)}%</span>
        </div>
      )}
      <div className="h-1.5 w-full rounded-full bg-white/5 overflow-hidden">
        <div
          className="h-full rounded-full conf-bar"
          style={{ width: `${value * 100}%`, background: color, boxShadow: `0 0 8px ${color}` }}
        />
      </div>
    </div>
  );
}

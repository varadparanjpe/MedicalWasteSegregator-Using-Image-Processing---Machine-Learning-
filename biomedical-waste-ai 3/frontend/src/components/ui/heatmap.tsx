import { useMemo } from "react";
import { CATEGORY_LABEL, cn } from "@/lib/utils";

interface Props {
  labels: string[];
  matrix: number[][];
  className?: string;
}

/**
 * Dark-theme confusion-matrix heatmap.
 * Colors cells via a viridis-like scale; diagonals highlighted green.
 */
export function ConfusionHeatmap({ labels, matrix, className }: Props) {
  const { max, row_sums } = useMemo(() => {
    let m = 0;
    const sums = matrix.map((row) => row.reduce((s, v) => s + v, 0));
    for (const row of matrix) for (const v of row) if (v > m) m = v;
    return { max: Math.max(m, 1), row_sums: sums };
  }, [matrix]);

  const cellColor = (v: number, isDiag: boolean) => {
    const t = v / max;
    if (v === 0) return "rgba(255,255,255,0.02)";
    if (isDiag) {
      // green → emerald gradient for correct predictions
      const a = 0.15 + t * 0.85;
      return `rgba(16, 185, 129, ${a})`;
    }
    // viridis-ish purple→pink for errors
    const a = 0.15 + t * 0.85;
    return `rgba(236, 72, 153, ${a})`;
  };

  return (
    <div className={cn("overflow-x-auto", className)}>
      <div
        className="grid gap-1"
        style={{
          gridTemplateColumns: `auto repeat(${labels.length}, minmax(44px, 1fr))`,
          minWidth: 420,
        }}
      >
        {/* Header row */}
        <div />
        {labels.map((l) => (
          <div
            key={"h-" + l}
            className="text-[10px] text-foreground/40 font-medium text-center px-1 pb-2 truncate"
            title={CATEGORY_LABEL[l] ?? l}
          >
            {(CATEGORY_LABEL[l] ?? l).replace(" Waste", "").slice(0, 10)}
          </div>
        ))}

        {/* Data rows */}
        {matrix.map((row, i) => (
          <>
            {/* Row label */}
            <div
              key={"rl-" + i}
              className="text-[10px] text-foreground/40 font-medium text-right pr-2 flex items-center justify-end truncate"
              title={CATEGORY_LABEL[labels[i]] ?? labels[i]}
              style={{ maxWidth: 100 }}
            >
              {(CATEGORY_LABEL[labels[i]] ?? labels[i]).replace(" Waste", "").slice(0, 10)}
            </div>

            {/* Cells */}
            {row.map((v, j) => {
              const isDiag = i === j;
              const pct = row_sums[i] ? Math.round((v / row_sums[i]) * 100) : 0;
              return (
                <div
                  key={`c-${i}-${j}`}
                  className={cn(
                    "relative h-10 rounded-md flex items-center justify-center text-[11px] font-mono transition-transform hover:scale-110 cursor-default",
                    isDiag ? "ring-1 ring-emerald-500/30" : ""
                  )}
                  style={{
                    background: cellColor(v, isDiag),
                    color: v > max * 0.5 ? "#fff" : "rgba(255,255,255,0.75)",
                  }}
                  title={`true: ${labels[i]}  ·  pred: ${labels[j]}  ·  ${v} samples  ·  ${pct}% of row`}
                >
                  {v > 0 ? v : "·"}
                </div>
              );
            })}
          </>
        ))}
      </div>

      {/* Axis titles */}
      <div className="flex items-center justify-between mt-3 text-[10px] text-foreground/30 uppercase tracking-widest">
        <span>Rows: true class</span>
        <span>Cols: predicted class</span>
      </div>
    </div>
  );
}

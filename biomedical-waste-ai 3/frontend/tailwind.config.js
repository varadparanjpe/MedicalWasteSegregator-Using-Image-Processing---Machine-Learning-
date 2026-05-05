/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0c0414",
        surface: "#120c1e",
        foreground: "#e7e3ef",
        "foreground-muted": "#9ca3af",
        primary: { DEFAULT: "#a78bfa", foreground: "#0c0414" },
        muted: { DEFAULT: "#1c1428", foreground: "#9ca3af" },
        border: "rgba(255,255,255,0.08)",
        card: {
          DEFAULT: "rgba(255,255,255,0.03)",
          hover: "rgba(255,255,255,0.055)",
          foreground: "#e7e3ef",
        },
        /* Bin colours */
        "bin-white":  "#e5e7eb",
        "bin-yellow": "#f59e0b",
        "bin-red":    "#ef4444",
        "bin-blue":   "#3b82f6",
        "bin-black":  "#6b7280",
        /* Category palette */
        "cat-sharps":        "#ef4444",
        "cat-infectious":    "#f59e0b",
        "cat-pathological":  "#a855f7",
        "cat-plastic":       "#3b82f6",
        "cat-pharma":        "#10b981",
        "cat-general":       "#9ca3af",
        /* Status */
        hazard:   "#ef4444",
        warning:  "#f59e0b",
        success:  "#10b981",
      },
      fontFamily: {
        sans: ['"Inter var"', '"Inter"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      borderRadius: {
        "4xl": "2rem",
        "5xl": "2.5rem",
      },
      backgroundImage: {
        "hero-glow": `
          radial-gradient(ellipse 900px 600px at 50% -100px, rgba(139,92,246,0.28) 0%, transparent 70%),
          radial-gradient(ellipse 700px 400px at 10%  80%, rgba(59,130,246,0.18) 0%, transparent 70%),
          radial-gradient(ellipse 600px 400px at 90%  70%, rgba(236,72,153,0.15) 0%, transparent 70%)
        `,
        "card-glow":
          "radial-gradient(ellipse 300px 200px at 50% 0%, rgba(167,139,250,0.1), transparent 70%)",
        "gradient-primary":
          "linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #ec4899 100%)",
        "gradient-warm":
          "linear-gradient(135deg, #f59e0b 0%, #ef4444 100%)",
      },
      boxShadow: {
        glass:  "0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06)",
        card:   "0 8px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05)",
        "glow-primary": "0 0 40px rgba(139,92,246,0.4)",
        "glow-red":     "0 0 30px rgba(239,68,68,0.5)",
        "glow-amber":   "0 0 30px rgba(245,158,11,0.5)",
        "btn-primary":  "0 4px 20px rgba(139,92,246,0.4), 0 1px 3px rgba(0,0,0,0.3)",
      },
      animation: {
        "fade-in":    "fadeIn 0.45s ease-out both",
        "fade-up":    "fadeInUp 0.55s cubic-bezier(0.4,0,0.2,1) both",
        "orb":        "orb 8s ease-in-out infinite",
        "orb-slow":   "orb 12s ease-in-out infinite",
        "pulse-soft": "pulse 3s ease-in-out infinite",
        "shimmer":    "shimmer 2s ease-in-out infinite",
        "alert":      "alertPulse 2s ease-in-out infinite",
        "spin-slow":  "spin 8s linear infinite",
      },
      keyframes: {
        fadeIn:    { from: { opacity: "0" }, to: { opacity: "1" } },
        fadeInUp:  {
          from: { opacity: "0", transform: "translateY(20px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        orb: {
          "0%,100%": { transform: "scale(1) translateY(0px)" },
          "50%":     { transform: "scale(1.12) translateY(-18px)" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" },
        },
        alertPulse: {
          "0%,100%": { opacity: "1" },
          "50%":     { opacity: "0.55" },
        },
      },
      transitionTimingFunction: {
        spring: "cubic-bezier(0.34, 1.56, 0.64, 1)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

import { Link, useLocation } from "react-router-dom";
import { Activity, BarChart3, Home, Biohazard, Video, GitCompare } from "lucide-react";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";

const links = [
  { to: "/",          label: "Home",      icon: Home },
  { to: "/predict",   label: "Analyze",   icon: Activity },
  { to: "/realtime",  label: "Realtime",  icon: Video },
  { to: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { to: "/compare",   label: "Compare",   icon: GitCompare },
];

export function Navbar() {
  const { pathname } = useLocation();
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let alive = true;
    const check = async () => {
      try { const h = await getHealth(); if (alive) setOnline(h.pipeline); }
      catch { if (alive) setOnline(false); }
    };
    check();
    const t = setInterval(check, 10_000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  return (
    <nav className="fixed top-4 inset-x-0 z-50 flex justify-center pointer-events-none">
      <div className="glass-heavy rounded-full px-2 py-1.5 flex items-center gap-1 pointer-events-auto shadow-card">
        <Link to="/" className="flex items-center gap-2 pl-2 pr-3 mr-1 border-r border-white/10">
          <div className="h-6 w-6 rounded-md bg-gradient-to-br from-violet-600 to-fuchsia-600 flex items-center justify-center">
            <Biohazard className="h-3.5 w-3.5 text-white" />
          </div>
          <span className="text-xs font-semibold text-foreground/80 hidden sm:block">
            BioWaste AI
          </span>
        </Link>

        {links.map(({ to, label, icon: Icon }) => {
          const active = pathname === to;
          return (
            <Link
              key={to}
              to={to}
              className={cn(
                "flex items-center gap-1.5 px-3 py-2 rounded-full text-xs font-medium transition-all duration-200",
                active
                  ? "bg-gradient-to-r from-violet-600/40 to-fuchsia-600/40 text-white border border-white/10"
                  : "text-foreground/50 hover:text-foreground hover:bg-white/5"
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              <span className="hidden md:block">{label}</span>
            </Link>
          );
        })}

        <div className="ml-1 pl-3 border-l border-white/10 pr-1 flex items-center gap-1.5">
          <div className={cn(
            "h-2 w-2 rounded-full transition-colors duration-500",
            online === null  && "bg-white/20 animate-pulse",
            online === true  && "bg-emerald-400 shadow-[0_0_6px_2px_rgba(52,211,153,0.5)]",
            online === false && "bg-red-400 shadow-[0_0_6px_2px_rgba(239,68,68,0.5)]"
          )} />
          <span className="text-[10px] text-foreground/30 hidden md:block">
            {online === null ? "…" : online ? "API live" : "API offline"}
          </span>
        </div>
      </div>
    </nav>
  );
}

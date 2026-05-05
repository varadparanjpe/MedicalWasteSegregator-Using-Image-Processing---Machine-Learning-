import { lazy, Suspense } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Navbar } from "@/components/ui/navbar";
import { Loader2 } from "lucide-react";

const Landing         = lazy(() => import("@/pages/Landing"));
const Prediction      = lazy(() => import("@/pages/Prediction"));
const Realtime        = lazy(() => import("@/pages/Realtime"));
const Dashboard       = lazy(() => import("@/pages/Dashboard"));
const ModelComparison = lazy(() => import("@/pages/ModelComparison"));

function PageLoader() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0c0414]">
      <Loader2 className="h-8 w-8 text-violet-400 animate-spin" />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Suspense fallback={<PageLoader />}>
        <Routes>
          <Route path="/"          element={<Landing />} />
          <Route path="/predict"   element={<Prediction />} />
          <Route path="/realtime"  element={<Realtime />} />
          <Route path="/dashboard"   element={<Dashboard />} />
          <Route path="/compare"    element={<ModelComparison />} />
          <Route path="*" element={
            <div className="min-h-screen flex flex-col items-center justify-center bg-[#0c0414] text-foreground gap-4">
              <p className="text-6xl font-bold gradient-text">404</p>
              <p className="text-foreground/50">Page not found</p>
              <a href="/" className="text-violet-400 hover:underline text-sm">Go home</a>
            </div>
          } />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

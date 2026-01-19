import React, { useEffect, useState } from "react";
import { useNavigate }                 from "react-router-dom";
import { useMutation }                 from "@tanstack/react-query";
import { AnimatePresence, motion }     from "framer-motion";

import UploadZone                      from "./UploadZone";          // 🆕
import QuickViz                        from "../charts/QuickViz";
import { api }                         from "../api/client";
import { useWizard }                   from "../store/wizardStore";

/* ---------- backend types ---------- */
type ValResp = {
  valid: boolean;
  warnings: string[];
  suggestions: string[];
  preview: Record<string, unknown>[];
  columns: { name: string; type: string; unique: number; sample: string[] }[];
  plot_base64?: string;
  llm_insights?: string;
  server_path: string;
  model_catalog: any;
};

/* ======================================================================= */
export default function SetupPage() {
  const nav  = useNavigate();
  const wiz  = useWizard();
  const { useCase, suggestion, validation, set } = wiz;

  /* ------------------- backend mutations ------------------- */
  const suggestMut = useMutation({
    mutationFn : (txt: string) =>
      api.post("/suggest-task-model", { use_case: txt }).then(r => r.data),
    onSuccess  : (data) => set({ suggestion: data }),
  });

  const uploadMut = useMutation({
    mutationFn : (file: File) => {
      const fd = new FormData();
      fd.append("file", file);
      return api.post<ValResp>("/validate-dataset", fd).then(r => r.data);
    },
    onSuccess  : (res) => set({
      validation   : res,
      serverPath   : res.server_path,
      modelCatalog : res.model_catalog,
    }),
  });

  /* ------------------- helpers ------------------- */
  const done      = suggestion && validation?.valid;
  const [collapsed, setCollapsed] = useState(false);
  useEffect(() => { if (!collapsed && done) setCollapsed(true); }, [done]);

  /* ------------------- motion config ------------------- */
  const HEADER_H = 120;
  const heroVars = {
    open     : { y: "-50%", x: "-50%", scale: 1 },
    collapsed: { y: 0,      x: 0,      scale: 0.9 },
  };

  /* ============================= UI ============================= */
  return (
    <main className="h-screen w-full overflow-hidden bg-white">
      {/* ---------- HERO ---------- */}
      <motion.div
        variants={heroVars}
        animate={collapsed ? "collapsed" : "open"}
        transition={{ type: "spring", stiffness: 80 }}
        style={{
          position   : collapsed ? "fixed" : "absolute",
          top        : collapsed ? 0       : "50%",
          left       : collapsed ? 0       : "50%",
          width      : "100%",
          zIndex     : 40,
          padding    : "1.5rem",
          background : collapsed ? "#ffffff" : "transparent",
          boxShadow  : collapsed ? "0 2px 6px rgba(0,0,0,0.06)" : "none",
          height     : collapsed ? HEADER_H : "auto",
          display    : "flex",
          flexDirection: collapsed ? "row" : "column",
          alignItems   : "center",
          gap          : "1rem",
        }}
      >
        {/* use-case textarea */}
        <textarea
          className={`border rounded shadow focus:outline-none p-3 ${
            collapsed ? "flex-1 max-w-[30rem] h-10" : "w-full max-w-xl"
          }`}
          placeholder="Describe your NLP task (e.g. classify tweets by sentiment)…"
          value={useCase}
          onChange={(e) => set({ useCase: e.target.value })}
          rows={collapsed ? 1 : 3}
        />

        {/* drag-and-drop zone  */}
        <UploadZone
          disabled={!!validation}
          onFile={(f) => uploadMut.mutate(f)}
        />

        {/* analyse button */}
        <button
          className={`rounded text-white transition-colors ${
            collapsed
              ? "px-4 py-2 bg-blue-500"
              : "w-full max-w-xl py-2 bg-blue-600"
          } disabled:opacity-40`}
          disabled={!useCase || !validation || suggestMut.isLoading}
          onClick={() => suggestMut.mutate(useCase)}
        >
          {suggestMut.isLoading ? "Analysing…" : "Analyse Task + Dataset ➜"}
        </button>
      </motion.div>

      {/* ---------- ANALYTICS AREA ---------- */}
      <AnimatePresence>
        {collapsed && validation && (
          <motion.div
            key="analytics"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{ paddingTop: HEADER_H, height: "100%", overflowY: "auto" }}
            className="px-6 pb-6 space-y-6"
          >
            {/* warnings & suggestions */}
            <div className="text-sm space-y-1">
              {validation.warnings.map((w, i) => (
                <p key={i} className="text-yellow-700">⚠️ {w}</p>
              ))}
              {validation.suggestions.map((s, i) => (
                <p key={i} className="text-blue-700">💡 {s}</p>
              ))}
            </div>

            {/* label plot */}
            {validation.plot_base64 && (
              <img
                src={`data:image/png;base64,${validation.plot_base64}`}
                alt="label distribution"
                className="max-w-md border rounded shadow"
              />
            )}

            {/* column statistics & quick-viz  (unchanged) */}
            {/* … */}
          </motion.div>
        )}
      </AnimatePresence>

      {/* ---------- FOOTER ---------- */}
      <div className="p-4 border-t bg-white text-right fixed bottom-0 left-0 right-0">
        <button
          disabled={!done}
          onClick={() => nav("/model")}
          className="px-6 py-2 bg-teal-600 text-white rounded disabled:opacity-40"
        >
          Next: Select Model →
        </button>
      </div>
    </main>
  );
}

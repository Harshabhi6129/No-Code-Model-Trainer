import React, { useEffect, useState, useRef } from "react";
import { useNavigate }                       from "react-router-dom";
import { useQuery, useMutation }             from "@tanstack/react-query";
import Form                                  from "@rjsf/core";
import validator                             from "@rjsf/validator-ajv8";
import type { JSONSchema7 }                  from "json-schema";
import type { FieldTemplateProps }           from "@rjsf/utils";

import { api }       from "../api/client";
import { useWizard } from "../store/wizardStore";
import LiveChart     from "../charts/LiveChart";
import ChatWidget    from "./ChatWidget";                 // 🆕

/* ---------- FieldTemplate ---------- */
const FieldRow: React.FC<FieldTemplateProps> = ({
  id, label, required, description, errors, children, classNames,
}) => (
  <div className={`grid grid-cols-5 gap-3 items-start ${classNames}`}>
    <div className="col-span-3">
      <label htmlFor={id} className="font-medium">
        {label}
        {required && <span className="text-red-500">*</span>}
      </label>
      {description}
      {errors}
    </div>
    <div className="col-span-2">{children}</div>
  </div>
);

const uiSchema = { "ui:options": { label: true } };

/* ================================================================= */
export default function ModelTrainPage() {
  const nav  = useNavigate();
  const wiz  = useWizard();
  const {
    suggestion, validation, serverPath, modelCatalog,
    runId, hparams, paramSchema,
    metrics, logs, commentary, wandbUrl,
    pushLog, pushMetric, pushCommentary, pushChat,   // 🆕
    clearLogs, set,
  } = wiz;

  /* ---------- redirect guard ---------- */
  useEffect(() => {
    if (!suggestion || !validation) nav("/setup", { replace: true });
  }, [suggestion, validation, nav]);

  /* ---------- local state ---------- */
  const [modelId,  setModelId]  = useState(suggestion?.recommended_model || "");
  const [usePeft,  setUsePeft]  = useState(false);
  const [formData, setFormData] = useState<any>(hparams || {});
  const [status,   setStatus]   = useState<"idle"|"running"|"completed"|"errored">("idle");

  /* ---------- fetch param schema ---------- */
  const schemaQ = useQuery<JSONSchema7>({
    queryKey   : ["param-schema", modelId],
    queryFn    : () => api.get("/model-params", { params: { model_id: modelId } }).then(r => r.data),
    enabled    : !!modelId,
    staleTime  : 300_000,
    initialData: paramSchema,
  });

  /* ---------- fetch recommended defaults (Phase 2) ---------- */
  useQuery({
    queryKey : ["param-suggest", modelId, validation?.row_count],
    queryFn  : () => api.post("/suggest-hparams", {
      model_id: modelId,
      stats   : validation,
    }).then(r => r.data),
    enabled  : !!modelId && !!validation,
    onSuccess: (defaults) => {
      setFormData((prev) => ({ ...defaults, ...prev }));
      set({ hparams: defaults });
    },
  });

  /* ---------- WebSocket ---------- */
  const wsRef = useRef<WebSocket|null>(null);            // 🆕

  const openWS = (rid: string) => {
    clearLogs();
    const base = new URL(api.defaults.baseURL || "http://localhost:8000");
    const ws   = new WebSocket(
      `${base.protocol === "https:" ? "wss" : "ws"}//${base.host}/ws/train/${rid}`,
    );
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if      (msg.event === "log")        pushLog(msg.msg);
      else if (msg.event === "metric")     pushMetric(msg);
      else if (msg.event === "commentary") pushCommentary(msg.text);
      else if (msg.event === "wandb_url")  set({ wandbUrl: msg.url });
      else if (msg.event === "chat")       pushChat({ from: "agent", text: msg.text });             // 🆕
      else if (msg.event === "confusion") {                                                        // 🆕
        pushChat({ from: "agent", text: "(Confusion matrix below)" });
        pushChat({ from: "agent", text: `<img src="data:image/png;base64,${msg.img}" />` });
      }
      else if (msg.event === "status")     setStatus(msg.state);
    };
  };
  useEffect(() => { if (runId) openWS(runId); }, []);

  /* ---------- training mutation ---------- */
  const trainMut = useMutation({
    mutationFn: () =>
      api.post("/train", {
        dataset_path: serverPath,
        model       : modelId,
        task        : suggestion?.task,
        config      : { ...formData, use_peft: usePeft },
      }).then(r => r.data),
    onSuccess : ({ run_id }) => {
      set({
        runId     : run_id,
        hparams   : formData,
        paramSchema: schemaQ.data,
        wandbUrl  : undefined,
      });
      setStatus("running");
      openWS(run_id);
    },
  });

  /* ---------- helpers ---------- */
  const opts          = [...new Set(modelCatalog?.transformers || [])].sort();
  const isRunning     = status === "running";
  const isCompleted   = status === "completed";
  const disableInputs = isRunning || isCompleted;

  /* ============================  UI  ============================ */
  return (
    <main className="flex h-screen overflow-hidden">
      {/* ─── LEFT: controls ───────────────────────────────────────── */}
      <aside className="basis-1/3 min-w-[18rem] max-w-[24rem] border-r p-6 space-y-6 overflow-y-auto">
        {/* Model dropdown */}
        <div>
          <label className="block mb-1 font-medium">Model</label>
          <select
            disabled={disableInputs}
            className="w-full border p-2 rounded"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
          >
            <option value="" disabled>Select model…</option>
            {opts.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>

        {/* LoRA checkbox */}
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            disabled={disableInputs}
            checked={usePeft}
            onChange={(e) => setUsePeft(e.target.checked)}
          />
          Use LoRA adapters
        </label>

        {/* Param form */}
        {schemaQ.isSuccess && (
          <Form
            disabled      ={disableInputs}
            schema        ={schemaQ.data as JSONSchema7}
            formData      ={formData}
            onChange      ={(e) => setFormData(e.formData)}
            validator     ={validator}
            FieldTemplate ={FieldRow}
            uiSchema      ={uiSchema}
            className     ="space-y-4"
          >
            <></>
          </Form>
        )}

        {/* Buttons */}
        {!isRunning && !isCompleted && (
          <button
            disabled={!schemaQ.isSuccess || trainMut.isLoading}
            onClick={() => trainMut.mutate()}
            className="w-full py-2 bg-teal-600 text-white rounded disabled:opacity-40"
          >
            {trainMut.isLoading ? "Starting…" : "Start Training →"}
          </button>
        )}
        {isRunning && (
          <button
            onClick={() => { setStatus("idle"); set({ runId: "", wandbUrl: undefined }); clearLogs(); }}
            className="w-full py-2 bg-red-600 text-white rounded"
          >
            Stop Training ■
          </button>
        )}
        {isCompleted && (
          <>
            <button
              onClick={() => trainMut.mutate()}
              className="w-full py-2 bg-yellow-600 text-white rounded"
            >
              Restart Training ↻
            </button>
            <a
              href={`${api.defaults.baseURL}/export/${runId}`}
              className="block text-center mt-2 px-4 py-2 bg-indigo-600 text-white rounded"
            >
              Download Trained Model
            </a>
          </>
        )}
      </aside>

      {/* ─── RIGHT: dashboard + chat ──────────────────────────────── */}
      <section className="flex-1 flex overflow-hidden">
        {/* Dashboard column */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* W&B iframe */}
          {wandbUrl && (
            <div className="h-64 border-b">
              <iframe
                src={`${wandbUrl}?embed=true`}
                className="w-full h-full"
                title="Weights & Biases Live"
              />
            </div>
          )}

          {/* Live chart */}
          <div className="h-56 p-4">
            <LiveChart />
          </div>

          {/* Metrics + commentary */}
          <div className="flex-1 overflow-y-auto px-4">
            <table className="w-full text-xs border">
              <thead className="bg-gray-100 sticky top-0">
                <tr>
                  <th className="border px-1 py-0.5">Epoch</th>
                  <th className="border px-1">Loss</th>
                  <th className="border px-1">Val Loss</th>
                  <th className="border px-1">Val Acc</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m) => (
                  <tr key={m.epoch}>
                    <td className="border px-1 text-center">{m.epoch}</td>
                    <td className="border px-1">{m.loss?.toFixed(4)}</td>
                    <td className="border px-1">{m.val_loss?.toFixed(4)}</td>
                    <td className="border px-1">
                      {m.val_acc != null ? `${(m.val_acc * 100).toFixed(2)}%` : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="mt-4 space-y-1 text-sm">
              {commentary.map((c, i) => (
                <p
                  key={i}
                  className="bg-indigo-50 border-l-4 border-indigo-300 p-2 rounded"
                >
                  {c}
                </p>
              ))}
            </div>

            <pre className="mt-4 whitespace-pre-wrap text-xs border p-2 bg-gray-50 h-32 overflow-y-auto">
              {logs.join("\n")}
            </pre>
          </div>
        </div>

        {/* Chat sidebar */}
        <ChatWidget
          disabled={!wsRef.current}
          send={(txt) =>
            wsRef.current?.send(JSON.stringify({ event: "command", text: txt }))
          }
        />
      </section>
    </main>
  );
}

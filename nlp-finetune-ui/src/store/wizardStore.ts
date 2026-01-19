import { create } from "zustand";
import type { SuggestResp, ValidateResp } from "../api/types";

export interface WizardState {
  /* step-1 */
  useCase: string;
  suggestion?: SuggestResp;
  validation?: ValidateResp;
  serverPath?: string;
  modelCatalog?: {
    transformers: string[];
    finetuned: string[];
    advice: string;
  };

  /* step-2 */
  hparams?: Record<string, any>;
  paramSchema?: Record<string, any>;
  runId?: string;
  wandbUrl?: string;

  /* live data */
  logs: string[];
  metrics: any[];
  commentary: string[];

  /* chat */
  chat: { from: "user" | "agent"; text: string }[];

  /* actions */
  set:  (p: Partial<WizardState>) => void;
  pushLog:        (l: string) => void;
  pushMetric:     (m: any)    => void;
  pushCommentary: (t: string) => void;
  pushChat:       (m: {from:"user"|"agent";text:string}) => void;
  clearLogs:      ()          => void;
}

export const useWizard = create<WizardState>((set) => ({
  useCase: "",
  logs: [],
  metrics: [],
  commentary: [],
  chat: [],

  set: (p) => set(p),

  pushLog:        (l) => set((s) => ({ logs:        [...s.logs, l] })),
  pushMetric:     (m) => set((s) => ({ metrics:     [...s.metrics, m] })),
  pushCommentary: (t) => set((s) => ({ commentary: [...s.commentary, t] })),
  pushChat:       (m) => set((s) => ({ chat:        [...s.chat, m] })),

  clearLogs: () => set({ logs: [], metrics: [], commentary: [] }),
}));

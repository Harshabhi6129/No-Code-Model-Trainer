import { rest, setupWorker } from "msw";
import { v4 as uuid } from "uuid";
import { Server as MockSocketServer } from "mock-socket";
import type { SuggestResp, ValidateResp, HParamResp, TrainResp } from "./types";

/* utility */
const delay = (ms = 400) => new Promise(r => setTimeout(r, ms));

function makeFakeSocket(runId: string) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${protocol}//${window.location.host}/ws/train/${runId}`;

  const mock = new MockSocketServer(url);

  mock.on("connection", socket => {
    const lines = [
      "Epoch 1/3 - loss: 0.45 - acc: 0.76",
      "Epoch 2/3 - loss: 0.31 - acc: 0.85",
      "Epoch 3/3 - loss: 0.25 - acc: 0.90",
      "✔️ Training completed.",
    ];

    lines.forEach((ln, i) =>
      setTimeout(() => socket.send(ln), (i + 1) * 800),
    );

    setTimeout(() => socket.close(), (lines.length + 1) * 800);
  });
}

/* …all the other REST handlers stay the same… */


export const handlers = [
  rest.post("/suggest-task-model", async (req, res, ctx) => {
    await delay();
    const { use_case } = await req.json();
    const data: SuggestResp = {
      task: "spam_classification",
      recommended_model: "distilbert-base-uncased",
      reason: "DistilBERT is efficient for binary tasks like spam detection.",
    };
    return res(ctx.json(data));
  }),

  rest.post("/validate-dataset", async (_req, res, ctx) => {
    await delay();
    const data: ValidateResp = {
      valid: true,
      warnings: ["Class imbalance detected (80% spam)"],
      suggestions: ["Consider oversampling minority class"],
    };
    return res(ctx.json(data));
  }),

  rest.post("/suggest-hparams", async (_req, res, ctx) => {
    await delay();
    const data: HParamResp = {
      epochs: 3,
      batch_size: 8,
      learning_rate: 2e-5,
      use_peft: true,
    };
    return res(ctx.json(data));
  }),

  rest.post("/train", async (_req, res, ctx) => {
    await delay();
    const run_id = uuid();
    makeFakeSocket(run_id);
    const data: TrainResp = { run_id, status: "started" };
    return res(ctx.json(data));
  }),
];

export const worker = setupWorker(...handlers);

// src/charts/LiveChart.tsx

import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
} from "chart.js";
import { useWizard } from "../store/wizardStore";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  Legend
);

export default function LiveChart() {
  // subscribe to your store’s metrics array
  const metrics = useWizard((s) => s.metrics);

  if (metrics.length === 0) {
    return (
      <div className="text-center text-gray-500 py-6">
        Waiting for training to start…
      </div>
    );
  }

  const labels = metrics.map((m) => m.epoch.toString());

  const datasets = [
    {
      label: "Train Loss",
      data: metrics.map((m) => m.loss),
      borderColor: "#14b8a6",
      borderWidth: 2,
      tension: 0.3,
      yAxisID: "loss",
    },
    metrics.some((m) => m.val_loss != null) && {
      label: "Val Loss",
      data: metrics.map((m) => m.val_loss ?? null),
      borderColor: "#6366f1",
      borderWidth: 2,
      tension: 0.3,
      yAxisID: "loss",
    },
    metrics.some((m) => m.val_acc != null) && {
      label: "Val Acc (%)",
      data: metrics.map((m) =>
        m.val_acc != null ? (m.val_acc * 100).toFixed(1) : null
      ),
      borderColor: "#f97316",
      borderWidth: 2,
      tension: 0.3,
      yAxisID: "acc",
    },
  ].filter(Boolean);

  return (
    <Line
      data={{
        labels,
        datasets,
      }}
      options={{
        responsive: true,
        interaction: { mode: "index", intersect: false },
        stacked: false,
        plugins: {
          legend: { position: "top" },
          title: { display: true, text: "Live Training Metrics" },
        },
        scales: {
          x: {
            title: { display: true, text: "Epoch" },
          },
          loss: {
            type: "linear" as const,
            display: true,
            position: "left" as const,
            title: { display: true, text: "Loss" },
          },
          acc: {
            type: "linear" as const,
            display: datasets.some((d) => d.yAxisID === "acc"),
            position: "right" as const,
            title: { display: true, text: "Accuracy (%)" },
            grid: { drawOnChartArea: false },
          },
        },
      }}
    />
  );
}

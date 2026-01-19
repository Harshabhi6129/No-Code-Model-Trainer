import React, { useMemo } from "react";
import { Bar, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement);

export interface QuickVizProps {
  col: string;
  type: "numeric" | "categorical" | "text";
  values: (string | number)[];
}

export default function QuickViz({ col, type, values }: QuickVizProps) {
  const { labels, data } = useMemo(() => {
    if (type === "numeric" || type === "text") {
      /* histogram (10 bins) */
      const nums =
        type === "numeric"
          ? (values as number[])
          : (values as string[]).map((v) => v.length);
      const min = Math.min(...nums);
      const max = Math.max(...nums);
      const bins = 10;
      const counts = Array(bins).fill(0);
      nums.forEach((n) => {
        const idx = Math.min(bins - 1, Math.floor(((n - min) / (max - min)) * bins));
        counts[idx] += 1;
      });
      const bucketLabels = counts.map((_, i) => {
        const lo = min + ((max - min) / bins) * i;
        const hi = min + ((max - min) / bins) * (i + 1);
        return `${lo.toFixed(1)}-${hi.toFixed(1)}`;
      });
      return { labels: bucketLabels, data: counts };
    } else {
      /* categorical frequency */
      const freq: Record<string, number> = {};
      (values as string[]).forEach((v) => (freq[v] = (freq[v] || 0) + 1));
      return { labels: Object.keys(freq), data: Object.values(freq) };
    }
  }, [type, values]);

  const chartProps = {
    data: {
      labels,
      datasets: [{ data, label: col }],
    },
    options: {
      plugins: { legend: { display: false } },
      responsive: true,
      maintainAspectRatio: false,
    },
    height: 250,
  };

  return (
    <div className="min-w-[250px] w-full h-[260px]">
      {type === "numeric" || type === "text" ? (
        <Bar {...chartProps} />
      ) : (
        <Bar {...chartProps} />
      )}
    </div>
  );
}

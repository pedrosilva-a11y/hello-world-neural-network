import Plot from "react-plotly.js";
import type { Config, Data, Layout } from "plotly.js";

import {
  comparisonMetricDefinitions,
  getComparisonChartPoints,
  type ComparisonMetric,
} from "../../lib/comparisonChartData";
import type { NormalizedExperiment } from "../../types/summary";

type ComparisonChartsProps = {
  experiments: NormalizedExperiment[];
};

type ComparisonChartProps = {
  experiments: NormalizedExperiment[];
  metric: ComparisonMetric;
};

const chartConfig: Partial<Config> = {
  displayModeBar: false,
  responsive: true,
};

function truncateLabel(value: string, maxLength = 34): string {
  if (value.length <= maxLength) {
    return value;
  }

  return `${value.slice(0, maxLength - 1)}…`;
}

function getChartHeight(pointCount: number): number {
  return Math.max(320, pointCount * 34 + 120);
}

function ComparisonChart({ experiments, metric }: ComparisonChartProps) {
  const definition = comparisonMetricDefinitions[metric];
  const points = getComparisonChartPoints(experiments, metric);
  const chartHeight = getChartHeight(points.length);

  if (points.length === 0) {
    return (
      <article className="chart-card">
        <h3>{definition.title}</h3>
        <p className="muted">{definition.emptyMessage}</p>
      </article>
    );
  }

  const chartData: Data[] = [
    {
      type: "bar",
      orientation: "h",
      x: points.map((point) => point.value),
      y: points.map((point, index) =>
        `${index + 1}. ${truncateLabel(point.experimentName)}`,
      ),
      text: points.map((point) => point.formattedValue),
      customdata: points.map((point) => point.experimentName),
      textposition: "auto",
      hovertemplate: "%{customdata}<br>%{text}<extra></extra>",
    },
  ];

  const layout: Partial<Layout> = {
    autosize: true,
    height: chartHeight,
    margin: {
      l: 190,
      r: 24,
      t: 12,
      b: 52,
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: {
      title: {
        text: definition.xAxisTitle,
      },
      zeroline: false,
    },
    yaxis: {
      automargin: true,
      autorange: "reversed",
    },
  };

  return (
    <article className="chart-card">
      <div className="chart-card-header">
        <h3>{definition.title}</h3>
        <p className="muted">{definition.description}</p>
      </div>

      <div className="chart-plot" style={{ height: `${chartHeight}px` }}>
        <Plot
          config={chartConfig}
          data={chartData}
          layout={layout}
          style={{ height: `${chartHeight}px`, width: "100%" }}
          useResizeHandler
        />
      </div>
    </article>
  );
}

export function ComparisonCharts({ experiments }: ComparisonChartsProps) {
  return (
    <section className="dashboard-section">
      <div className="section-header">
        <div>
          <h2>Comparison charts</h2>
          <p className="muted">
            Charts update with the active filters. Showing up to 20 experiments
            per chart.
          </p>
        </div>
      </div>

      <div className="charts-grid">
        <ComparisonChart experiments={experiments} metric="validationAccuracy" />
        <ComparisonChart experiments={experiments} metric="validationLoss" />
        <ComparisonChart experiments={experiments} metric="validationErrorPercent" />
        <ComparisonChart experiments={experiments} metric="trainValidationGapPercent" />
      </div>
    </section>
  );
}

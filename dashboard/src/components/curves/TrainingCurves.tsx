import { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Config, Data, Layout } from "plotly.js";

import {
  getTrainingCurveExperiments,
  getTrainingCurveSeries,
  getValidationLossMinimumPoint,
  type CurveLineKind,
  type CurveMetric,
} from "../../lib/trainingCurveData";
import type { NormalizedExperiment } from "../../types/summary";

type TrainingCurvesProps = {
  experiments: NormalizedExperiment[];
};

type CurveLineMode = "both" | "train" | "validation";

type CurveChartProps = {
  experiments: NormalizedExperiment[];
  includedKinds: CurveLineKind[];
  maxIteration: number;
  maxPoints: number;
  metric: CurveMetric;
  useLogXAxis: boolean;
};

const chartConfig: Partial<Config> = {
  displayModeBar: false,
  responsive: true,
};

const maxPointOptions = [250, 500, 1000, 2000, 5000];

function getMetricTitle(metric: CurveMetric): string {
  return metric === "accuracy" ? "Training accuracy curves" : "Training loss curves";
}

function getMetricDescription(metric: CurveMetric): string {
  if (metric === "accuracy") {
    return "Train and validation accuracy over training iterations.";
  }

  return "Train and validation loss over training iterations. Markers show validation-loss minima.";
}

function getYAxisTitle(metric: CurveMetric): string {
  return metric === "accuracy" ? "Accuracy (%)" : "Loss";
}

function getIncludedKinds(lineMode: CurveLineMode): CurveLineKind[] {
  if (lineMode === "train") {
    return ["train"];
  }

  if (lineMode === "validation") {
    return ["validation"];
  }

  return ["train", "validation"];
}

function getMaximumIteration(experiments: NormalizedExperiment[]): number {
  const maximumIteration = experiments.reduce((currentMaximum, experiment) => {
    return Math.max(
      currentMaximum,
      experiment.trainLoss.length,
      experiment.validationLoss.length,
      experiment.trainAccuracy.length,
      experiment.validationAccuracy.length,
    );
  }, 1);

  return Math.max(1, maximumIteration);
}

function clampIteration(value: number, maximumAvailableIteration: number): number {
  if (!Number.isFinite(value)) {
    return maximumAvailableIteration;
  }

  return Math.min(Math.max(Math.round(value), 1), maximumAvailableIteration);
}

function buildCurveTraces(
  experiments: NormalizedExperiment[],
  metric: CurveMetric,
  maxPoints: number,
  maxIteration: number,
  includedKinds: CurveLineKind[],
): Data[] {
  const seriesList = getTrainingCurveSeries(
    experiments,
    metric,
    maxPoints,
    10,
    maxIteration,
    includedKinds,
  );

  return seriesList.map((series) => ({
    type: "scatter",
    mode: "lines",
    name: series.label,
    x: series.points.map((point) => point.x),
    y: series.points.map((point) => point.y),
    line: {
      dash: series.kind === "validation" ? "dash" : "solid",
    },
    hovertemplate: `${series.experimentName}<br>${series.kind}<br>iteration %{x}<br>%{y}<extra></extra>`,
  }));
}

function buildValidationLossMinimumTraces(
  experiments: NormalizedExperiment[],
  maxIteration: number,
): Data[] {
  return getTrainingCurveExperiments(experiments).flatMap((experiment) => {
    const point = getValidationLossMinimumPoint(experiment, maxIteration);

    if (point === null) {
      return [];
    }

    return [
      {
        type: "scatter",
        mode: "markers",
        name: `${experiment.experimentName} · min validation loss`,
        x: [point.x],
        y: [point.y],
        showlegend: false,
        hovertemplate: `${experiment.experimentName}<br>min validation loss<br>iteration %{x}<br>%{y}<extra></extra>`,
      },
    ];
  });
}

function CurveChart({
  experiments,
  includedKinds,
  maxIteration,
  maxPoints,
  metric,
  useLogXAxis,
}: CurveChartProps) {
  const traces = buildCurveTraces(
    experiments,
    metric,
    maxPoints,
    maxIteration,
    includedKinds,
  );

  const shouldShowValidationMarkers =
    metric === "loss" && includedKinds.includes("validation");

  const markerTraces = shouldShowValidationMarkers
    ? buildValidationLossMinimumTraces(experiments, maxIteration)
    : [];

  const data = [...traces, ...markerTraces];

  if (data.length === 0) {
    return (
      <article className="chart-card">
        <h3>{getMetricTitle(metric)}</h3>
        <p className="muted">No {metric} curve data available.</p>
      </article>
    );
  }

  const layout: Partial<Layout> = {
    autosize: true,
    height: 460,
    margin: {
      l: 64,
      r: 24,
      t: 16,
      b: 58,
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: {
      title: {
        text: "Iteration",
      },
      type: useLogXAxis ? "log" : "linear",
      zeroline: false,
    },
    yaxis: {
      title: {
        text: getYAxisTitle(metric),
      },
      zeroline: false,
    },
    legend: {
      orientation: "h",
    },
  };

  return (
    <article className="chart-card">
      <div className="chart-card-header">
        <h3>{getMetricTitle(metric)}</h3>
        <p className="muted">{getMetricDescription(metric)}</p>
      </div>

      <div className="chart-plot chart-plot-tall">
        <Plot
          config={chartConfig}
          data={data}
          layout={layout}
          style={{ height: "460px", width: "100%" }}
          useResizeHandler
        />
      </div>
    </article>
  );
}

export function TrainingCurves({ experiments }: TrainingCurvesProps) {
  const maximumAvailableIteration = useMemo(
    () => getMaximumIteration(experiments),
    [experiments],
  );

  const [lineMode, setLineMode] = useState<CurveLineMode>("both");
  const [maxIteration, setMaxIteration] = useState(maximumAvailableIteration);
  const [maxPoints, setMaxPoints] = useState(500);
  const [useLogXAxis, setUseLogXAxis] = useState(false);

  const effectiveMaxIteration = clampIteration(
    maxIteration,
    maximumAvailableIteration,
  );

  const includedKinds = getIncludedKinds(lineMode);

  function updateMaxIteration(value: number) {
    setMaxIteration(clampIteration(value, maximumAvailableIteration));
  }

  return (
    <section className="dashboard-section">
      <div className="section-header section-header-with-controls">
        <div>
          <h2>Training curves</h2>
          <p className="muted">
            Showing the top 10 filtered experiments by validation accuracy.
          </p>
        </div>

        <div className="curve-controls">
          <label className="range-control">
            Max iteration shown
            <input
              type="range"
              min="1"
              max={maximumAvailableIteration}
              value={effectiveMaxIteration}
              onChange={(event) => updateMaxIteration(Number(event.target.value))}
            />
            <span className="range-row">
              <input
                aria-label="Editable max iteration shown"
                className="iteration-number-input"
                type="number"
                min="1"
                max={maximumAvailableIteration}
                value={effectiveMaxIteration}
                onChange={(event) =>
                  updateMaxIteration(Number(event.target.value))
                }
              />
              <span className="range-value">
                / {maximumAvailableIteration.toLocaleString()}
              </span>
            </span>
          </label>

          <div className="segmented-control" aria-label="Curve line visibility">
            <button
              className={lineMode === "train" ? "segmented-button active" : "segmented-button"}
              type="button"
              onClick={() => setLineMode("train")}
            >
              Training
            </button>
            <button
              className={
                lineMode === "validation" ? "segmented-button active" : "segmented-button"
              }
              type="button"
              onClick={() => setLineMode("validation")}
            >
              Validation
            </button>
            <button
              className={lineMode === "both" ? "segmented-button active" : "segmented-button"}
              type="button"
              onClick={() => setLineMode("both")}
            >
              Both
            </button>
          </div>

          <label>
            Max plotted points
            <select
              value={maxPoints}
              onChange={(event) => setMaxPoints(Number(event.target.value))}
            >
              {maxPointOptions.map((option) => (
                <option key={option} value={option}>
                  {option.toLocaleString()}
                </option>
              ))}
            </select>
          </label>

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={useLogXAxis}
              onChange={(event) => setUseLogXAxis(event.target.checked)}
            />
            Log x-axis
          </label>
        </div>
      </div>

      <div className="curves-grid">
        <CurveChart
          experiments={experiments}
          includedKinds={includedKinds}
          maxIteration={effectiveMaxIteration}
          maxPoints={maxPoints}
          metric="accuracy"
          useLogXAxis={useLogXAxis}
        />
        <CurveChart
          experiments={experiments}
          includedKinds={includedKinds}
          maxIteration={effectiveMaxIteration}
          maxPoints={maxPoints}
          metric="loss"
          useLogXAxis={useLogXAxis}
        />
      </div>
    </section>
  );
}

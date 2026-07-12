import { downsampleSeries, type SeriesPoint } from "./downsample";
import type { NormalizedExperiment } from "../types/summary";

export type CurveMetric = "accuracy" | "loss";
export type CurveLineKind = "train" | "validation";

export type TrainingCurveSeries = {
  experimentName: string;
  label: string;
  kind: CurveLineKind;
  points: SeriesPoint[];
};

function isKnownNumber(value: number): boolean {
  return Number.isFinite(value);
}

function getExperimentSortValue(experiment: NormalizedExperiment): number {
  return experiment.finalValidationAccuracy ?? Number.NEGATIVE_INFINITY;
}

export function getTrainingCurveExperiments(
  experiments: NormalizedExperiment[],
  maxExperiments = 10,
): NormalizedExperiment[] {
  return [...experiments]
    .sort((left, right) => getExperimentSortValue(right) - getExperimentSortValue(left))
    .slice(0, maxExperiments);
}

function buildSeriesPoints(
  values: number[],
  metric: CurveMetric,
  maxIteration: number | null,
): SeriesPoint[] {
  return values
    .filter(isKnownNumber)
    .map((value, index) => ({
      x: index + 1,
      y: metric === "accuracy" ? value * 100 : value,
    }))
    .filter((point) => maxIteration === null || point.x <= maxIteration);
}

function getMetricValues(
  experiment: NormalizedExperiment,
  metric: CurveMetric,
  kind: CurveLineKind,
): number[] {
  if (metric === "accuracy" && kind === "train") {
    return experiment.trainAccuracy;
  }

  if (metric === "accuracy" && kind === "validation") {
    return experiment.validationAccuracy;
  }

  if (metric === "loss" && kind === "train") {
    return experiment.trainLoss;
  }

  return experiment.validationLoss;
}

export function getTrainingCurveSeries(
  experiments: NormalizedExperiment[],
  metric: CurveMetric,
  maxPoints: number,
  maxExperiments = 10,
  maxIteration: number | null = null,
  includedKinds: CurveLineKind[] = ["train", "validation"],
): TrainingCurveSeries[] {
  const selectedExperiments = getTrainingCurveExperiments(
    experiments,
    maxExperiments,
  );

  const series: TrainingCurveSeries[] = [];

  for (const experiment of selectedExperiments) {
    for (const kind of includedKinds) {
      const values = getMetricValues(experiment, metric, kind);
      const points = buildSeriesPoints(values, metric, maxIteration);

      if (points.length === 0) {
        continue;
      }

      series.push({
        experimentName: experiment.experimentName,
        label: `${experiment.experimentName} · ${kind}`,
        kind,
        points: downsampleSeries(points, maxPoints),
      });
    }
  }

  return series;
}

export function getValidationLossMinimumPoint(
  experiment: NormalizedExperiment,
  maxIteration: number | null = null,
): SeriesPoint | null {
  const validationLoss =
    maxIteration === null
      ? experiment.validationLoss
      : experiment.validationLoss.slice(0, maxIteration);

  if (validationLoss.length === 0) {
    return null;
  }

  let minimumIndex = 0;
  let minimumValue = validationLoss[0];

  for (let index = 1; index < validationLoss.length; index += 1) {
    const value = validationLoss[index];

    if (value < minimumValue) {
      minimumIndex = index;
      minimumValue = value;
    }
  }

  return {
    x: minimumIndex + 1,
    y: minimumValue,
  };
}

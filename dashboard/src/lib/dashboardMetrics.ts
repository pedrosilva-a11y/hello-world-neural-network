import type { NormalizedExperiment } from "../types/summary";

export type DashboardMetrics = {
  experimentCount: number;
  bestValidationAccuracy: number | null;
  bestValidationAccuracyExperimentName: string | null;
  bestValidationLoss: number | null;
  bestValidationLossExperimentName: string | null;
  lowestValidationErrorPercent: number | null;
  lowestValidationErrorExperimentName: string | null;
  averageValidationAccuracy: number | null;
};

function getAverage(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }

  const total = values.reduce((sum, value) => sum + value, 0);

  return total / values.length;
}

function isKnownMetric(value: number | null): value is number {
  return value !== null && Number.isFinite(value);
}

export function getDashboardMetrics(
  experiments: NormalizedExperiment[],
): DashboardMetrics {
  let bestValidationAccuracy: number | null = null;
  let bestValidationAccuracyExperimentName: string | null = null;

  let bestValidationLoss: number | null = null;
  let bestValidationLossExperimentName: string | null = null;

  let lowestValidationErrorPercent: number | null = null;
  let lowestValidationErrorExperimentName: string | null = null;

  const validationAccuracies: number[] = [];

  for (const experiment of experiments) {
    if (isKnownMetric(experiment.finalValidationAccuracy)) {
      validationAccuracies.push(experiment.finalValidationAccuracy);

      if (
        bestValidationAccuracy === null ||
        experiment.finalValidationAccuracy > bestValidationAccuracy
      ) {
        bestValidationAccuracy = experiment.finalValidationAccuracy;
        bestValidationAccuracyExperimentName = experiment.experimentName;
      }
    }

    if (
      isKnownMetric(experiment.finalValidationLoss) &&
      (bestValidationLoss === null ||
        experiment.finalValidationLoss < bestValidationLoss)
    ) {
      bestValidationLoss = experiment.finalValidationLoss;
      bestValidationLossExperimentName = experiment.experimentName;
    }

    if (
      isKnownMetric(experiment.validationErrorPercent) &&
      (lowestValidationErrorPercent === null ||
        experiment.validationErrorPercent < lowestValidationErrorPercent)
    ) {
      lowestValidationErrorPercent = experiment.validationErrorPercent;
      lowestValidationErrorExperimentName = experiment.experimentName;
    }
  }

  return {
    experimentCount: experiments.length,
    bestValidationAccuracy,
    bestValidationAccuracyExperimentName,
    bestValidationLoss,
    bestValidationLossExperimentName,
    lowestValidationErrorPercent,
    lowestValidationErrorExperimentName,
    averageValidationAccuracy: getAverage(validationAccuracies),
  };
}

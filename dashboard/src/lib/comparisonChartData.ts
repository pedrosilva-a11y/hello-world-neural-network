import type { NormalizedExperiment } from "../types/summary";

export type ComparisonMetric =
  | "validationAccuracy"
  | "validationLoss"
  | "validationErrorPercent"
  | "trainValidationGapPercent";

export type ComparisonChartPoint = {
  experimentName: string;
  value: number;
  formattedValue: string;
};

export type ComparisonMetricDefinition = {
  title: string;
  description: string;
  xAxisTitle: string;
  emptyMessage: string;
  sortDirection: "ascending" | "descending";
  getValue: (experiment: NormalizedExperiment) => number | null;
  formatValue: (value: number) => string;
};

export const comparisonMetricDefinitions: Record<
  ComparisonMetric,
  ComparisonMetricDefinition
> = {
  validationAccuracy: {
    title: "Validation accuracy",
    description: "Higher is better. Values are final validation accuracy.",
    xAxisTitle: "Accuracy (%)",
    emptyMessage: "No validation accuracy values available.",
    sortDirection: "descending",
    getValue: (experiment) =>
      experiment.finalValidationAccuracy === null
        ? null
        : experiment.finalValidationAccuracy * 100,
    formatValue: (value) => `${value.toFixed(2)}%`,
  },
  validationLoss: {
    title: "Validation loss",
    description: "Lower is better. Values are final validation loss.",
    xAxisTitle: "Loss",
    emptyMessage: "No validation loss values available.",
    sortDirection: "ascending",
    getValue: (experiment) => experiment.finalValidationLoss,
    formatValue: (value) => value.toFixed(4),
  },
  validationErrorPercent: {
    title: "Validation error",
    description: "Lower is better. Computed as 100 × (1 - validation accuracy).",
    xAxisTitle: "Error (%)",
    emptyMessage: "No validation error values available.",
    sortDirection: "ascending",
    getValue: (experiment) => experiment.validationErrorPercent,
    formatValue: (value) => `${value.toFixed(2)}%`,
  },
  trainValidationGapPercent: {
    title: "Train-validation gap",
    description:
      "Higher values can indicate overfitting pressure. Computed from final accuracies.",
    xAxisTitle: "Gap (percentage points)",
    emptyMessage: "No train-validation gap values available.",
    sortDirection: "descending",
    getValue: (experiment) => experiment.trainValidationGapPercent,
    formatValue: (value) => `${value.toFixed(2)} pp`,
  },
};

function isKnownMetric(value: number | null): value is number {
  return value !== null && Number.isFinite(value);
}

function compareChartPoints(
  left: ComparisonChartPoint,
  right: ComparisonChartPoint,
  sortDirection: "ascending" | "descending",
): number {
  if (sortDirection === "ascending") {
    return left.value - right.value;
  }

  return right.value - left.value;
}

export function getComparisonChartPoints(
  experiments: NormalizedExperiment[],
  metric: ComparisonMetric,
  maxExperiments = 20,
): ComparisonChartPoint[] {
  const definition = comparisonMetricDefinitions[metric];

  return experiments
    .map((experiment) => {
      const value = definition.getValue(experiment);

      if (!isKnownMetric(value)) {
        return null;
      }

      return {
        experimentName: experiment.experimentName,
        value,
        formattedValue: definition.formatValue(value),
      };
    })
    .filter((point): point is ComparisonChartPoint => point !== null)
    .sort((left, right) =>
      compareChartPoints(left, right, definition.sortDirection),
    )
    .slice(0, maxExperiments);
}

import type { NormalizedExperiment } from "../types/summary";
import { formatNumber } from "./formatters";

export type DashboardFilters = {
  modelNames: string[];
  architectures: string[];
  normalizationOptions: string[];
  optimizers: string[];
  learningRateKeys: string[];
  iterationCountKeys: string[];
  experimentNames: string[];
};

export type FilterOption = {
  value: string;
  label: string;
  count: number;
};

export const emptyDashboardFilters: DashboardFilters = {
  modelNames: [],
  architectures: [],
  normalizationOptions: [],
  optimizers: [],
  learningRateKeys: [],
  iterationCountKeys: [],
  experimentNames: [],
};

function getNormalizationKey(experiment: NormalizedExperiment): string {
  if (experiment.normalizePixels === null) {
    return "unknown";
  }

  return experiment.normalizePixels ? "normalized" : "not_normalized";
}

function getNormalizationLabel(value: string): string {
  if (value === "normalized") {
    return "Normalized";
  }

  if (value === "not_normalized") {
    return "Not normalized";
  }

  return "Unknown";
}

function getExperimentFilterValue(
  experiment: NormalizedExperiment,
  filterKey: keyof DashboardFilters,
): string {
  switch (filterKey) {
    case "modelNames":
      return experiment.modelName;
    case "architectures":
      return experiment.neuronsProfileLabel;
    case "normalizationOptions":
      return getNormalizationKey(experiment);
    case "optimizers":
      return experiment.optimizer;
    case "learningRateKeys":
      return experiment.learningRateKey;
    case "iterationCountKeys":
      return experiment.numIterationsKey;
    case "experimentNames":
      return experiment.experimentName;
  }
}

function getOptionLabel(
  experiment: NormalizedExperiment,
  filterKey: keyof DashboardFilters,
): string {
  switch (filterKey) {
    case "normalizationOptions":
      return getNormalizationLabel(getNormalizationKey(experiment));
    case "learningRateKeys":
      return formatNumber(experiment.learningRate);
    case "iterationCountKeys":
      return formatNumber(experiment.numIterations);
    default:
      return getExperimentFilterValue(experiment, filterKey);
  }
}

function selectedFilterAllowsValue(selectedValues: string[], value: string): boolean {
  if (selectedValues.length === 0) {
    return true;
  }

  return selectedValues.includes(value);
}

export function filterExperiments(
  experiments: NormalizedExperiment[],
  filters: DashboardFilters,
): NormalizedExperiment[] {
  return experiments.filter((experiment) => {
    return (
      selectedFilterAllowsValue(filters.modelNames, experiment.modelName) &&
      selectedFilterAllowsValue(filters.architectures, experiment.neuronsProfileLabel) &&
      selectedFilterAllowsValue(
        filters.normalizationOptions,
        getNormalizationKey(experiment),
      ) &&
      selectedFilterAllowsValue(filters.optimizers, experiment.optimizer) &&
      selectedFilterAllowsValue(filters.learningRateKeys, experiment.learningRateKey) &&
      selectedFilterAllowsValue(filters.iterationCountKeys, experiment.numIterationsKey) &&
      selectedFilterAllowsValue(filters.experimentNames, experiment.experimentName)
    );
  });
}

export function getFilterOptions(
  experiments: NormalizedExperiment[],
  filterKey: keyof DashboardFilters,
): FilterOption[] {
  const countsByValue = new Map<string, number>();
  const labelsByValue = new Map<string, string>();

  for (const experiment of experiments) {
    const value = getExperimentFilterValue(experiment, filterKey);
    const label = getOptionLabel(experiment, filterKey);

    countsByValue.set(value, (countsByValue.get(value) ?? 0) + 1);
    labelsByValue.set(value, label);
  }

  return [...countsByValue.entries()]
    .map(([value, count]) => ({
      value,
      label: labelsByValue.get(value) ?? value,
      count,
    }))
    .sort((left, right) => left.label.localeCompare(right.label));
}

export function hasActiveFilters(filters: DashboardFilters): boolean {
  return Object.values(filters).some((selectedValues) => selectedValues.length > 0);
}

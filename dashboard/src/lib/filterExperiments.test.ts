import { describe, expect, it } from "vitest";

import {
  emptyDashboardFilters,
  filterExperiments,
  getFilterOptions,
  hasActiveFilters,
  type DashboardFilters,
} from "./filterExperiments";
import type { NormalizedExperiment } from "../types/summary";

function createExperiment(
  overrides: Partial<NormalizedExperiment>,
): NormalizedExperiment {
  return {
    experimentName: "experiment",
    modelName: "unknown_model",
    neuronsProfile: [],
    neuronsProfileLabel: "Unknown",
    normalizePixels: null,
    optimizer: "unknown",
    learningRate: null,
    learningRateKey: "unknown",
    numIterations: null,
    numIterationsKey: "unknown",
    trainLoss: [],
    validationLoss: [],
    trainAccuracy: [],
    validationAccuracy: [],
    finalTrainLoss: null,
    finalValidationLoss: null,
    finalTrainAccuracy: null,
    finalValidationAccuracy: null,
    validationErrorPercent: null,
    trainValidationGapPercent: null,
    modifiedAt: null,
    raw: {},
    ...overrides,
  };
}

const experiments = [
  createExperiment({
    experimentName: "softmax_baseline",
    modelName: "single_layer_softmax_classifier",
    neuronsProfileLabel: "[10]",
    normalizePixels: false,
    optimizer: "batch_gradient_descent",
    learningRate: 0.1,
    learningRateKey: "0.1",
    numIterations: 500,
    numIterationsKey: "500",
  }),
  createExperiment({
    experimentName: "relu_128_normalized_lr_01_1k",
    modelName: "one_hidden_layer_relu_classifier",
    neuronsProfileLabel: "[128, 10]",
    normalizePixels: true,
    optimizer: "batch_gradient_descent",
    learningRate: 0.1,
    learningRateKey: "0.1",
    numIterations: 1000,
    numIterationsKey: "1000",
  }),
  createExperiment({
    experimentName: "relu_128_128_normalized_lr_01_5k",
    modelName: "multi_layer_relu_classifier",
    neuronsProfileLabel: "[128, 128, 10]",
    normalizePixels: true,
    optimizer: "batch_gradient_descent",
    learningRate: 0.1,
    learningRateKey: "0.1",
    numIterations: 5000,
    numIterationsKey: "5000",
  }),
  createExperiment({
    experimentName: "unknown_legacy_experiment",
    modelName: "unknown_model",
    neuronsProfileLabel: "Unknown",
    normalizePixels: null,
    optimizer: "unknown",
    learningRate: null,
    learningRateKey: "unknown",
    numIterations: null,
    numIterationsKey: "unknown",
  }),
];

describe("filterExperiments", () => {
  it("returns all experiments when filters are empty", () => {
    expect(filterExperiments(experiments, emptyDashboardFilters)).toEqual(
      experiments,
    );
  });

  it("filters by model name", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      modelNames: ["multi_layer_relu_classifier"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([experiments[2]]);
  });

  it("filters by architecture", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      architectures: ["[128, 128, 10]"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([experiments[2]]);
  });

  it("filters by normalized experiments", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      normalizationOptions: ["normalized"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([
      experiments[1],
      experiments[2],
    ]);
  });

  it("filters by not-normalized experiments", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      normalizationOptions: ["not_normalized"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([experiments[0]]);
  });

  it("filters by unknown normalization", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      normalizationOptions: ["unknown"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([experiments[3]]);
  });

  it("filters by optimizer", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      optimizers: ["batch_gradient_descent"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([
      experiments[0],
      experiments[1],
      experiments[2],
    ]);
  });

  it("filters by learning-rate key", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      learningRateKeys: ["0.1"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([
      experiments[0],
      experiments[1],
      experiments[2],
    ]);
  });

  it("filters by unknown learning-rate key", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      learningRateKeys: ["unknown"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([experiments[3]]);
  });

  it("filters by iteration-count key", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      iterationCountKeys: ["5000"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([experiments[2]]);
  });

  it("filters by unknown iteration-count key", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      iterationCountKeys: ["unknown"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([experiments[3]]);
  });

  it("filters by experiment name", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      experimentNames: ["relu_128_128_normalized_lr_01_5k"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([experiments[2]]);
  });

  it("combines multiple active filters with AND behavior", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      modelNames: ["multi_layer_relu_classifier"],
      normalizationOptions: ["normalized"],
      iterationCountKeys: ["5000"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([experiments[2]]);
  });

  it("returns no experiments when active filters conflict", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      modelNames: ["multi_layer_relu_classifier"],
      normalizationOptions: ["not_normalized"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([]);
  });
});

describe("getFilterOptions", () => {
  it("returns no options when there are no experiments", () => {
    expect(getFilterOptions([], "modelNames")).toEqual([]);
  });

  it("builds counted model options sorted by label", () => {
    expect(getFilterOptions(experiments, "modelNames")).toEqual([
      {
        value: "multi_layer_relu_classifier",
        label: "multi_layer_relu_classifier",
        count: 1,
      },
      {
        value: "one_hidden_layer_relu_classifier",
        label: "one_hidden_layer_relu_classifier",
        count: 1,
      },
      {
        value: "single_layer_softmax_classifier",
        label: "single_layer_softmax_classifier",
        count: 1,
      },
      {
        value: "unknown_model",
        label: "unknown_model",
        count: 1,
      },
    ]);
  });

  it("builds architecture options", () => {
    expect(getFilterOptions(experiments, "architectures")).toEqual([
      {
        value: "[10]",
        label: "[10]",
        count: 1,
      },
      {
        value: "[128, 10]",
        label: "[128, 10]",
        count: 1,
      },
      {
        value: "[128, 128, 10]",
        label: "[128, 128, 10]",
        count: 1,
      },
      {
        value: "Unknown",
        label: "Unknown",
        count: 1,
      },
    ]);
  });

  it("builds normalization options with human-readable labels", () => {
    expect(getFilterOptions(experiments, "normalizationOptions")).toEqual([
      {
        value: "normalized",
        label: "Normalized",
        count: 2,
      },
      {
        value: "not_normalized",
        label: "Not normalized",
        count: 1,
      },
      {
        value: "unknown",
        label: "Unknown",
        count: 1,
      },
    ]);
  });

  it("builds optimizer options", () => {
    expect(getFilterOptions(experiments, "optimizers")).toEqual([
      {
        value: "batch_gradient_descent",
        label: "batch_gradient_descent",
        count: 3,
      },
      {
        value: "unknown",
        label: "unknown",
        count: 1,
      },
    ]);
  });

  it("builds learning-rate options using formatted labels", () => {
    expect(getFilterOptions(experiments, "learningRateKeys")).toEqual([
      {
        value: "0.1",
        label: "0.1",
        count: 3,
      },
      {
        value: "unknown",
        label: "Unknown",
        count: 1,
      },
    ]);
  });

  it("builds iteration-count options using formatted labels", () => {
    expect(getFilterOptions(experiments, "iterationCountKeys")).toEqual([
      {
        value: "1000",
        label: "1,000",
        count: 1,
      },
      {
        value: "5000",
        label: "5,000",
        count: 1,
      },
      {
        value: "500",
        label: "500",
        count: 1,
      },
      {
        value: "unknown",
        label: "Unknown",
        count: 1,
      },
    ]);
  });

  it("builds experiment-name options", () => {
    expect(getFilterOptions(experiments, "experimentNames")).toEqual([
      {
        value: "relu_128_128_normalized_lr_01_5k",
        label: "relu_128_128_normalized_lr_01_5k",
        count: 1,
      },
      {
        value: "relu_128_normalized_lr_01_1k",
        label: "relu_128_normalized_lr_01_1k",
        count: 1,
      },
      {
        value: "softmax_baseline",
        label: "softmax_baseline",
        count: 1,
      },
      {
        value: "unknown_legacy_experiment",
        label: "unknown_legacy_experiment",
        count: 1,
      },
    ]);
  });
});

describe("hasActiveFilters", () => {
  it("returns false when all filters are empty", () => {
    expect(hasActiveFilters(emptyDashboardFilters)).toBe(false);
  });

  it.each<keyof DashboardFilters>([
    "modelNames",
    "architectures",
    "normalizationOptions",
    "optimizers",
    "learningRateKeys",
    "iterationCountKeys",
    "experimentNames",
  ])("returns true when %s has a selected value", (filterKey) => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      [filterKey]: ["selected"],
    };

    expect(hasActiveFilters(filters)).toBe(true);
  });
});

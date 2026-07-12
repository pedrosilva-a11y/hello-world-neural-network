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

describe("filterExperiments", () => {
  const experiments = [
    createExperiment({
      experimentName: "softmax_baseline",
      modelName: "single_layer_softmax_classifier",
      neuronsProfileLabel: "[10]",
      normalizePixels: false,
      optimizer: "gradient_descent",
      learningRate: 0.1,
      learningRateKey: "0.1",
      numIterations: 500,
      numIterationsKey: "500",
    }),
    createExperiment({
      experimentName: "relu_128_128_normalized_lr_01_5k",
      modelName: "multi_layer_relu_classifier",
      neuronsProfileLabel: "[128, 128, 10]",
      normalizePixels: true,
      optimizer: "gradient_descent",
      learningRate: 0.1,
      learningRateKey: "0.1",
      numIterations: 5000,
      numIterationsKey: "5000",
    }),
  ];

  it("returns all experiments when filters are empty", () => {
    expect(filterExperiments(experiments, emptyDashboardFilters)).toHaveLength(2);
  });

  it("filters by architecture", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      architectures: ["[128, 128, 10]"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([experiments[1]]);
  });

  it("filters by normalization option", () => {
    const filters: DashboardFilters = {
      ...emptyDashboardFilters,
      normalizationOptions: ["normalized"],
    };

    expect(filterExperiments(experiments, filters)).toEqual([experiments[1]]);
  });

  it("builds counted filter options", () => {
    const options = getFilterOptions(experiments, "learningRateKeys");

    expect(options).toEqual([
      {
        value: "0.1",
        label: "0.1",
        count: 2,
      },
    ]);
  });

  it("detects active filters", () => {
    expect(hasActiveFilters(emptyDashboardFilters)).toBe(false);

    expect(
      hasActiveFilters({
        ...emptyDashboardFilters,
        modelNames: ["multi_layer_relu_classifier"],
      }),
    ).toBe(true);
  });
});

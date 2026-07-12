import { describe, expect, it } from "vitest";

import { getChangedConfigDiffRows, getConfigDiffRows } from "./configDiff";

describe("config diff", () => {
  it("detects changed nested config values", () => {
    const leftConfig = {
      model: {
        name: "single_layer_softmax_classifier",
        neurons_profile: [10],
      },
      training: {
        learning_rate: 0.1,
        num_iterations: 500,
      },
    };

    const rightConfig = {
      model: {
        name: "multi_layer_relu_classifier",
        neurons_profile: [128, 128, 10],
      },
      training: {
        learning_rate: 0.1,
        num_iterations: 5000,
      },
    };

    expect(getChangedConfigDiffRows(leftConfig, rightConfig)).toEqual([
      {
        path: "model.name",
        leftValue: "single_layer_softmax_classifier",
        rightValue: "multi_layer_relu_classifier",
        isDifferent: true,
      },
      {
        path: "model.neurons_profile",
        leftValue: "[10]",
        rightValue: "[128,128,10]",
        isDifferent: true,
      },
      {
        path: "training.num_iterations",
        leftValue: "500",
        rightValue: "5000",
        isDifferent: true,
      },
    ]);
  });

  it("shows missing values when a key exists on only one side", () => {
    const rows = getChangedConfigDiffRows(
      {
        preprocessing: {
          normalize_pixels: true,
        },
      },
      {},
    );

    expect(rows).toEqual([
      {
        path: "preprocessing.normalize_pixels",
        leftValue: "true",
        rightValue: "Missing",
        isDifferent: true,
      },
    ]);
  });

  it("can return unchanged rows too", () => {
    const rows = getConfigDiffRows(
      {
        training: {
          learning_rate: 0.1,
        },
      },
      {
        training: {
          learning_rate: 0.1,
        },
      },
    );

    expect(rows).toEqual([
      {
        path: "training.learning_rate",
        leftValue: "0.1",
        rightValue: "0.1",
        isDifferent: false,
      },
    ]);
  });
});

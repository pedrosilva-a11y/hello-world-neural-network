import { afterEach, describe, expect, it, vi } from "vitest";

import { buildCsv, downloadCsv } from "./csv";

describe("csv helpers", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns an empty string when there are no rows", () => {
    expect(buildCsv([])).toBe("");
  });

  it("builds CSV content and escapes special values", () => {
    const csv = buildCsv([
      {
        experiment_name: "baseline",
        notes: 'uses "plain" softmax',
        accuracy: 0.91,
      },
      {
        experiment_name: "relu,normalized",
        notes: "contains comma",
        accuracy: 0.97,
      },
    ]);

    expect(csv).toBe(
      'experiment_name,notes,accuracy\nbaseline,"uses ""plain"" softmax",0.91\n"relu,normalized",contains comma,0.97\n',
    );
  });

  it("creates and clicks a temporary download link", () => {
    const createObjectUrlSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:mock-url");

    const revokeObjectUrlSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => undefined);

    const originalCreateElement = document.createElement.bind(document);
    const clickSpy = vi.fn();

    vi.spyOn(document, "createElement").mockImplementation((tagName) => {
      if (tagName !== "a") {
        return originalCreateElement(tagName);
      }

      const anchor = originalCreateElement("a");
      anchor.click = clickSpy;

      return anchor;
    });

    downloadCsv("experiments.csv", "a,b\n");

    expect(createObjectUrlSpy).toHaveBeenCalledOnce();
    expect(clickSpy).toHaveBeenCalledOnce();
    expect(revokeObjectUrlSpy).toHaveBeenCalledWith("blob:mock-url");
  });
});

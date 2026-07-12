export type ConfigDiffRow = {
  path: string;
  leftValue: string;
  rightValue: string;
  isDifferent: boolean;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function serializeValue(value: unknown): string {
  if (value === undefined) {
    return "__missing__";
  }

  return JSON.stringify(value);
}

function formatValue(value: unknown): string {
  if (value === undefined) {
    return "Missing";
  }

  if (typeof value === "string") {
    return value;
  }

  if (
    value === null ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return String(value);
  }

  return JSON.stringify(value);
}

function flattenObject(
  value: unknown,
  prefix = "",
  flattened = new Map<string, unknown>(),
): Map<string, unknown> {
  if (!isRecord(value)) {
    if (prefix !== "") {
      flattened.set(prefix, value);
    }

    return flattened;
  }

  const entries = Object.entries(value);

  if (entries.length === 0 && prefix !== "") {
    flattened.set(prefix, {});
    return flattened;
  }

  for (const [key, childValue] of entries) {
    const childPath = prefix === "" ? key : `${prefix}.${key}`;

    if (isRecord(childValue)) {
      flattenObject(childValue, childPath, flattened);
    } else {
      flattened.set(childPath, childValue);
    }
  }

  return flattened;
}

export function getConfigDiffRows(
  leftConfig: unknown,
  rightConfig: unknown,
): ConfigDiffRow[] {
  const leftValues = flattenObject(leftConfig);
  const rightValues = flattenObject(rightConfig);

  const paths = [...new Set([...leftValues.keys(), ...rightValues.keys()])].sort(
    (left, right) => left.localeCompare(right),
  );

  return paths.map((path) => {
    const leftValue = leftValues.get(path);
    const rightValue = rightValues.get(path);
    const isDifferent = serializeValue(leftValue) !== serializeValue(rightValue);

    return {
      path,
      leftValue: formatValue(leftValue),
      rightValue: formatValue(rightValue),
      isDifferent,
    };
  });
}

export function getChangedConfigDiffRows(
  leftConfig: unknown,
  rightConfig: unknown,
): ConfigDiffRow[] {
  return getConfigDiffRows(leftConfig, rightConfig).filter(
    (row) => row.isDifferent,
  );
}

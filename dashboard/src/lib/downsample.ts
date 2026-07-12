export type SeriesPoint = {
  x: number;
  y: number;
};

export function downsampleSeries(
  points: SeriesPoint[],
  maxPoints: number,
): SeriesPoint[] {
  if (maxPoints <= 0) {
    return [];
  }

  if (points.length <= maxPoints) {
    return points;
  }

  if (maxPoints === 1) {
    return [points[0]];
  }

  const sampledPoints: SeriesPoint[] = [];
  const usedIndexes = new Set<number>();
  const lastIndex = points.length - 1;

  for (let targetIndex = 0; targetIndex < maxPoints; targetIndex += 1) {
    const sourceIndex = Math.round((targetIndex * lastIndex) / (maxPoints - 1));

    if (!usedIndexes.has(sourceIndex)) {
      sampledPoints.push(points[sourceIndex]);
      usedIndexes.add(sourceIndex);
    }
  }

  if (!usedIndexes.has(lastIndex)) {
    sampledPoints.push(points[lastIndex]);
  }

  return sampledPoints;
}

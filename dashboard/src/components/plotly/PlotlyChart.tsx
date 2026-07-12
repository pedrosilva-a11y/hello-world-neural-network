import Plot from "react-plotly.js";

export type PlotlyChartData = Array<Record<string, unknown>>;
export type PlotlyChartLayout = Record<string, unknown>;
export type PlotlyChartConfig = Record<string, unknown>;

type PlotlyChartProps = {
  config: PlotlyChartConfig;
  data: PlotlyChartData;
  height: number;
  layout: PlotlyChartLayout;
};

export function PlotlyChart({
  config,
  data,
  height,
  layout,
}: PlotlyChartProps) {
  return (
    <Plot
      config={config}
      data={data}
      layout={layout}
      style={{ height: `${height}px`, width: "100%" }}
      useResizeHandler
    />
  );
}
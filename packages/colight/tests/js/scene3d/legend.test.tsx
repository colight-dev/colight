import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import React from "react";
import {
  collectLegendEntries,
  formatTick,
  gradientCss,
  legendReport,
  Legend,
  SceneLegends,
  ColorByMeta,
} from "../../../src/js/scene3d/legend";

const CONTINUOUS: ColorByMeta = {
  cmap: "viridis",
  categorical: false,
  label: "Cu %",
  domain: [0, 2.5],
  stops: [
    [0.267004, 0.004874, 0.329415],
    [0.993248, 0.906157, 0.143936],
  ],
};

const CATEGORICAL: ColorByMeta = {
  cmap: "tab10",
  categorical: true,
  label: "rock type",
  categories: ["ore", "waste"],
  colors: [
    [0.121569, 0.466667, 0.705882],
    [1.0, 0.498039, 0.054902],
  ],
};

describe("formatTick", () => {
  it("formats compact tick labels", () => {
    expect(formatTick(0)).toBe("0");
    expect(formatTick(2.5)).toBe("2.5");
    expect(formatTick(1.23456)).toBe("1.235");
    expect(formatTick(123456)).toBe("1.2e+5");
    expect(formatTick(0.0001)).toBe("1.0e-4");
  });
});

describe("gradientCss", () => {
  it("builds a linear-gradient through the stops", () => {
    const css = gradientCss(CONTINUOUS.stops!);
    expect(css).toContain("linear-gradient(to right");
    expect(css).toContain("rgb(68, 1, 84) 0.0%");
    expect(css).toContain("rgb(253, 231, 37) 100.0%");
  });
});

describe("collectLegendEntries", () => {
  it("collects color_by specs with compiled component indices", () => {
    const entries = collectLegendEntries([
      { type: "PointCloud" },
      { type: "Cuboid", color_by: CONTINUOUS },
      null,
      { type: "LineSegments", color_by: CATEGORICAL },
    ]);
    expect(entries).toHaveLength(2);
    expect(entries[0].component).toEqual({ index: 1, type: "Cuboid" });
    expect(entries[1].component).toEqual({ index: 3, type: "LineSegments" });
  });

  it("skips legends disabled via legend: false", () => {
    const entries = collectLegendEntries([
      { type: "Cuboid", color_by: { ...CONTINUOUS, legend: false } },
    ]);
    expect(entries).toHaveLength(0);
  });

  it("dedupes identical specs across components", () => {
    const entries = collectLegendEntries([
      { type: "Cuboid", color_by: CONTINUOUS },
      { type: "Cuboid", color_by: { ...CONTINUOUS } },
      { type: "Cuboid", color_by: { ...CONTINUOUS, label: "other" } },
    ]);
    expect(entries).toHaveLength(2);
    expect(entries[0].component.index).toBe(0);
    expect(entries[1].component.index).toBe(2);
  });
});

describe("legendReport", () => {
  it("is lean: no color tables, includes component identity", () => {
    const report = legendReport(CONTINUOUS, { index: 2, type: "Cuboid" });
    expect(report).toEqual({
      cmap: "viridis",
      categorical: false,
      label: "Cu %",
      domain: [0, 2.5],
      component: 2,
      type: "Cuboid",
    });
    expect(report).not.toHaveProperty("stops");
  });
});

describe("rendering", () => {
  it("renders a continuous legend with label, ticks and report attribute", () => {
    const { container } = render(<Legend spec={CONTINUOUS} />);
    expect(container.textContent).toContain("Cu %");
    expect(container.textContent).toContain("0");
    expect(container.textContent).toContain("1.25");
    expect(container.textContent).toContain("2.5");
    const card = container.querySelector("[data-colight-legend]")!;
    const report = JSON.parse(card.getAttribute("data-colight-legend")!);
    expect(report.cmap).toBe("viridis");
  });

  it("renders categorical swatches with category names", () => {
    const { container } = render(<Legend spec={CATEGORICAL} />);
    expect(container.textContent).toContain("rock type");
    expect(container.textContent).toContain("ore");
    expect(container.textContent).toContain("waste");
  });

  it("SceneLegends groups entries by dock position", () => {
    const { container } = render(
      <SceneLegends
        entries={[
          { spec: CONTINUOUS, component: { index: 0, type: "Cuboid" } },
          {
            spec: { ...CATEGORICAL, position: "bottom-left" },
            component: { index: 1, type: "PointCloud" },
          },
        ]}
      />,
    );
    const cards = container.querySelectorAll("[data-colight-legend]");
    expect(cards).toHaveLength(2);
  });

  it("SceneLegends renders nothing for empty entries", () => {
    const { container } = render(<SceneLegends entries={[]} />);
    expect(container.querySelectorAll("[data-colight-legend]")).toHaveLength(0);
  });
});

import { describe, it, expect } from "vitest";
import {
  colorizeChannel,
  channelValueAt,
  activeChannelName,
  applyActiveChannel,
  ColorChannel,
  ColorChannels,
} from "../../../src/js/scene3d/colorize";

// A 2-entry LUT: black -> white. domain [0, 10].
const continuousChannel: ColorChannel = {
  label: "Cu %",
  legend: { cmap: "viridis", categorical: false, domain: [0, 10] },
  colorizer: {
    kind: "continuous",
    lut: [
      [0, 0, 0],
      [1, 1, 1],
    ],
    domain: [0, 10],
  },
  values: [0, 5, 10, NaN],
  count: 4,
};

const categoricalChannel: ColorChannel = {
  label: "Lithology",
  legend: {
    cmap: "categorical",
    categorical: true,
    categories: [
      { value: 0, label: "not logged", color: [0.5, 0.5, 0.5] },
      { value: 1, label: "Dacite", color: [1, 0, 0] },
    ],
    fallback: { label: "unmapped", color: [0, 0, 1] },
  },
  colorizer: {
    kind: "categorical",
    categories: [
      { value: 0, label: "not logged", color: [0.5, 0.5, 0.5] },
      { value: 1, label: "Dacite", color: [1, 0, 0] },
    ],
    fallback: { label: "unmapped", color: [0, 0, 1] },
  },
  values: [0, 1, 9, NaN],
  count: 4,
};

describe("colorizeChannel — continuous", () => {
  it("samples the LUT across the domain with clamping", () => {
    const out = colorizeChannel(continuousChannel);
    // value 0 -> LUT[0] black; 10 -> LUT[last] white; 5 -> midpoint (nearest).
    expect([out[0], out[1], out[2]]).toEqual([0, 0, 0]);
    expect([out[6], out[7], out[8]]).toEqual([1, 1, 1]);
  });

  it("maps NaN to the nan color", () => {
    const out = colorizeChannel(continuousChannel);
    expect([out[9], out[10], out[11]]).toEqual([0.5, 0.5, 0.5]);
  });
});

describe("colorizeChannel — categorical", () => {
  it("maps declared codes and falls back for unmatched/NaN", () => {
    const out = colorizeChannel(categoricalChannel);
    expect([out[0], out[1], out[2]]).toEqual([0.5, 0.5, 0.5]); // code 0
    expect([out[3], out[4], out[5]]).toEqual([1, 0, 0]); // code 1
    expect([out[6], out[7], out[8]]).toEqual([0, 0, 1]); // code 9 -> fallback
    expect([out[9], out[10], out[11]]).toEqual([0, 0, 1]); // NaN -> fallback
  });
});

describe("channelValueAt", () => {
  it("returns raw numeric value for continuous channels", () => {
    expect(channelValueAt(continuousChannel, 1)).toBe(5);
    expect(channelValueAt(continuousChannel, 3)).toBeNull(); // NaN
  });

  it("returns the label for categorical channels", () => {
    expect(channelValueAt(categoricalChannel, 0)).toBe("not logged");
    expect(channelValueAt(categoricalChannel, 1)).toBe("Dacite");
    expect(channelValueAt(categoricalChannel, 2)).toBe("unmapped"); // fallback
  });
});

describe("activeChannelName", () => {
  const channels: ColorChannels = {
    CU_pct: continuousChannel,
    Lithology: categoricalChannel,
  };
  it("uses a matching name", () => {
    expect(activeChannelName(channels, "Lithology")).toBe("Lithology");
  });
  it("defaults to the first channel for unknown/undefined", () => {
    expect(activeChannelName(channels, undefined)).toBe("CU_pct");
    expect(activeChannelName(channels, "nope")).toBe("CU_pct");
  });
});

describe("applyActiveChannel", () => {
  it("rewrites colors + color_by for the active channel, no geometry", () => {
    const component: any = {
      type: "Cuboid",
      centers: new Float32Array([0, 0, 0, 1, 0, 0]),
      color_channels: {
        CU_pct: continuousChannel,
        Lithology: categoricalChannel,
      },
      active_channel: "Lithology",
    };
    const centersBefore = component.centers;
    const name = applyActiveChannel(component);
    expect(name).toBe("Lithology");
    expect(component._activeChannel).toBe("Lithology");
    expect(component.color_by).toBe(categoricalChannel.legend);
    expect(component.colors).toBeInstanceOf(Float32Array);
    // Geometry untouched (same reference) — a switch does not rebuild it.
    expect(component.centers).toBe(centersBefore);
  });

  it("is a no-op without color_channels", () => {
    const component: any = { type: "Cuboid", colors: new Float32Array([1]) };
    expect(applyActiveChannel(component)).toBeNull();
  });
});

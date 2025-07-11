import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import TopBar from "../../src/js/TopBar";

describe("TopBar", () => {
  const mockProps = {
    currentFile: "path/to/file.colight.py",
    connected: true,
    focusedPath: null,
    setFocusedPath: vi.fn(),
    browsingDirectory: false,
    setBrowsingDirectory: vi.fn(),
    isLoading: false,
    pragmaOverrides: {
      hideCode: false,
      hideProse: false,
    },
    setPragmaOverrides: vi.fn(),
  };

  it("should render breadcrumbs for current file", () => {
    render(<TopBar {...mockProps} />);
    expect(screen.getByText("root")).toBeTruthy();
    expect(screen.getByText("path")).toBeTruthy();
    expect(screen.getByText("to")).toBeTruthy();
    expect(screen.getByText("file.colight.py")).toBeTruthy();
  });

  it("should show connection status", () => {
    render(<TopBar {...mockProps} />);
    const indicator = screen.getByTitle("Connected");
    expect(indicator).toBeTruthy();
  });

  it("should show disconnected status", () => {
    render(<TopBar {...mockProps} connected={false} />);
    const indicator = screen.getByTitle("Disconnected");
    expect(indicator).toBeTruthy();
  });

  it("should show loading status", () => {
    render(<TopBar {...mockProps} isLoading={true} />);
    const indicator = screen.getByTitle("Loading...");
    expect(indicator).toBeTruthy();
  });

  it("should toggle code visibility", () => {
    render(<TopBar {...mockProps} />);
    const codeButton = screen.getByTitle("Code shown - click to hide");
    fireEvent.click(codeButton);
    expect(mockProps.setPragmaOverrides).toHaveBeenCalledWith(
      expect.any(Function),
    );
  });

  it("should handle root breadcrumb click", () => {
    render(<TopBar {...mockProps} />);
    fireEvent.click(screen.getByText("root"));
    expect(mockProps.setBrowsingDirectory).toHaveBeenCalledWith(true);
    expect(mockProps.setFocusedPath).toHaveBeenCalledWith(null);
  });
});

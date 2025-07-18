import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import CommandBar from "../../src/js/CommandBar";

describe("CommandBar", () => {
  const mockProps = {
    isOpen: true,
    onClose: vi.fn(),
    directoryTree: null,
    currentFile: null,
    onOpenFile: vi.fn(),
    pragmaOverrides: {
      hideCode: false,
      hideProse: false,
      hideStatements: false,
      hideVisuals: false,
    },
    setPragmaOverrides: vi.fn(),
    focusedPath: null,
    setFocusedPath: vi.fn(),
  };

  it("should not render when closed", () => {
    render(<CommandBar {...mockProps} isOpen={false} />);
    expect(screen.queryByPlaceholderText(/search files/i)).toBe(null);
  });

  it("should render when open", () => {
    render(<CommandBar {...mockProps} />);
    expect(screen.getByPlaceholderText(/search files/i)).toBeTruthy();
  });

  it("should close on Escape key", () => {
    render(<CommandBar {...mockProps} />);
    const input = screen.getByPlaceholderText(/search files/i);
    fireEvent.keyDown(input, { key: "Escape" });
    expect(mockProps.onClose).toHaveBeenCalled();
  });

  it("should close when clicking backdrop", () => {
    render(<CommandBar {...mockProps} />);
    // Click the backdrop (parent of the modal)
    const backdrop = document.querySelector(".fixed.inset-0");
    if (backdrop) fireEvent.click(backdrop);
    expect(mockProps.onClose).toHaveBeenCalled();
  });
});

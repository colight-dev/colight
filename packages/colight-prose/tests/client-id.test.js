import { describe, it, expect, beforeEach, vi } from "vitest";
import { getClientId, clearClientId } from "../src/js/client-id.js";

describe("Client ID Management", () => {
  beforeEach(() => {
    // Clear sessionStorage before each test
    clearClientId();
  });

  it("should generate a new client ID if none exists", () => {
    const clientId = getClientId();
    expect(clientId).toBeTruthy();
    expect(clientId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );
  });

  it("should return the same client ID on subsequent calls", () => {
    const clientId1 = getClientId();
    const clientId2 = getClientId();
    expect(clientId1).toBe(clientId2);
  });

  it("should persist client ID in sessionStorage", () => {
    const clientId = getClientId();
    const stored = sessionStorage.getItem("colight-live-client-id");
    expect(stored).toBe(clientId);
  });

  it("should generate a new ID after clearing", () => {
    const clientId1 = getClientId();
    clearClientId();
    const clientId2 = getClientId();
    expect(clientId1).not.toBe(clientId2);
  });

  it("should use crypto.randomUUID when available", () => {
    const mockUUID = "12345678-1234-4234-8234-123456789abc";
    const originalRandomUUID = global.crypto?.randomUUID;

    // Mock crypto.randomUUID
    if (global.crypto) {
      global.crypto.randomUUID = vi.fn(() => mockUUID);
    }

    clearClientId();
    const clientId = getClientId();

    expect(clientId).toBe(mockUUID);
    if (global.crypto) {
      expect(global.crypto.randomUUID).toHaveBeenCalled();
      // Restore
      global.crypto.randomUUID = originalRandomUUID;
    }
  });
});

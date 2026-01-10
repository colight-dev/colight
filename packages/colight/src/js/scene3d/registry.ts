/**
 * @module registry
 * @description Primitive registry for Scene3D extensibility.
 *
 * The registry allows custom primitives to be registered and used alongside built-in primitives.
 */

import { PrimitiveSpec } from "./types";
import {
  pointCloudSpec,
  ellipsoidSpec,
  ellipsoidAxesSpec,
  cuboidSpec,
  lineBeamsSpec,
  boundingBoxSpec,
} from "./primitives";

/**
 * Registry interface for managing primitive specs.
 */
export interface PrimitiveRegistry {
  /** Get a primitive spec by type name */
  get(type: string): PrimitiveSpec<any> | undefined;

  /** Register a primitive spec */
  set(type: string, spec: PrimitiveSpec<any>): void;

  /** Check if a primitive type is registered */
  has(type: string): boolean;

  /** Get all registered type names */
  types(): string[];

  /** Iterate over all entries */
  entries(): IterableIterator<[string, PrimitiveSpec<any>]>;
}

/**
 * Create a new primitive registry.
 *
 * @param seed - Optional initial primitives to register
 * @returns A new PrimitiveRegistry instance
 */
export function createPrimitiveRegistry(
  seed?: Record<string, PrimitiveSpec<any>>,
): PrimitiveRegistry {
  const specs = new Map<string, PrimitiveSpec<any>>();

  // Add seed primitives if provided
  if (seed) {
    for (const [type, spec] of Object.entries(seed)) {
      specs.set(type, spec);
    }
  }

  return {
    get(type: string): PrimitiveSpec<any> | undefined {
      return specs.get(type);
    },

    set(type: string, spec: PrimitiveSpec<any>): void {
      specs.set(type, spec);
    },

    has(type: string): boolean {
      return specs.has(type);
    },

    types(): string[] {
      return Array.from(specs.keys());
    },

    entries(): IterableIterator<[string, PrimitiveSpec<any>]> {
      return specs.entries();
    },
  };
}

/**
 * Built-in primitive specs.
 * These are registered by default in the default registry.
 */
export const builtInPrimitives: Record<string, PrimitiveSpec<any>> = {
  PointCloud: pointCloudSpec,
  Ellipsoid: ellipsoidSpec,
  EllipsoidAxes: ellipsoidAxesSpec,
  Cuboid: cuboidSpec,
  LineBeams: lineBeamsSpec,
  BoundingBox: boundingBoxSpec,
};

/**
 * The default shared primitive registry.
 * Contains all built-in primitives.
 */
export const defaultPrimitiveRegistry: PrimitiveRegistry =
  createPrimitiveRegistry(builtInPrimitives);

/**
 * Register a primitive in the default registry.
 *
 * @param type - The primitive type name
 * @param spec - The primitive spec
 */
export function registerPrimitive(
  type: string,
  spec: PrimitiveSpec<any>,
): void {
  defaultPrimitiveRegistry.set(type, spec);
}

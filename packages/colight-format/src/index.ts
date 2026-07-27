/**
 * A dependency-free JavaScript/TypeScript writer for the `.colight` container
 * format, version 2.
 *
 * Written from the specification in `docs/src/colight_docs/format.md` alone.
 * Nothing here imports from the Colight widget or any other Colight package, so
 * a Node producer can take this module standalone.
 *
 * ```ts
 * import { createFile, ndarray } from "@colight/format";
 *
 * const bytes = createFile(
 *   { ast: null, state: { points: ndarray(new Float32Array([0, 1, 2]), [3]) } },
 *   [{ ast: null, state: { frame: 1 } }],
 * );
 * ```
 *
 * Node users get file helpers from `@colight/format/node`.
 */

export {
  ALIGNMENT,
  CURRENT_VERSION,
  HEADER_SIZE,
  MAGIC_BYTES,
  align8,
} from "./constants.js";

export {
  DTYPE_BYTES,
  type Dtype,
  assertDtype,
  byteLengthFor,
  dtypeOfTypedArray,
  isDtype,
} from "./dtypes.js";

export {
  PyFloat,
  encodeJson,
  encodeJsonString,
  encodeNumber,
  pyFloat,
  type JsonValue,
} from "./json.js";

export {
  NDArray,
  RawBuffer,
  boolArray,
  ndarray,
  rawBuffer,
  type NDArraySpec,
} from "./values.js";

export {
  appendUpdates,
  assertAppendable,
  createEntry,
  createFile,
  createUpdateEntry,
  layoutBuffers,
  type BufferLayout,
  type Payload,
} from "./writer.js";

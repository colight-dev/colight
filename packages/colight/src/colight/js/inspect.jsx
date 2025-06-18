export function inspect({ value }) {
  // Handle simple cases
  if (typeof value === "number") {
    return `${value}`;
  }

  if (typeof value === "string") {
    return `"${value}"`;
  }

  if (typeof value === "boolean") {
    return `${value}`;
  }

  if (value === null) {
    return "null";
  }

  if (value === undefined) {
    return "undefined";
  }

  console.log(value);

  return "Inspected value";
}

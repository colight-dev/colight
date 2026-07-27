import { useState, useRef, useEffect } from "react";
import { tw } from "./utils";

// Color constants for primitive types
const PRIMITIVE_COLORS = {
  boolean: {
    true: "text-green-600",
    false: "text-red-600",
  },
  string: "text-green-600",
  number: "text-sky-500",
  null: "text-gray-500 italic",
  datetime: "font-mono",
};

// Bracket styles for different collection types
const BRACKET_STYLES = {
  list: { open: "[", close: "]" },
  tuple: { open: "(", close: ")" },
  set: { open: "{", close: "}" },
  dict: { open: "{", close: "}" },
};

function CompactExpandable({
  typeInfo,
  children,
  defaultExpanded = false,
  headerTag,
}) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [isHovered, setIsHovered] = useState(false);
  const brackets = BRACKET_STYLES[typeInfo.type] || { open: "(", close: ")" };
  const bracketClasses = `cursor-pointer font-mono text-gray-500 p-1 ${isHovered ? "bg-gray-100" : ""}`;

  return (
    <div
      className={tw("inline-block align-top")}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <span
        className={tw(
          `${bracketClasses} ${isExpanded ? "rounded-md" : "rounded-l-md"}`,
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className={tw("text-gray-600 select-none pl-1")}>
          {isExpanded ? "▼" : "▶"}
        </span>
        {headerTag && <span className={tw("ml-2 mr-1")}>{headerTag}</span>}
        <span className={tw(`ml-1`)}>{brackets.open}</span>
      </span>
      {isExpanded ? (
        <>
          <div className={tw("ml-8")}>{children}</div>
          <span
            className={tw(
              `${bracketClasses} ${isExpanded ? "rounded-md" : "rounded-r-md"} ml-5`,
            )}
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {brackets.close}
          </span>
        </>
      ) : (
        <span
          className={tw(
            `${bracketClasses} pl-0 ${isExpanded ? "rounded-md" : "rounded-r-md"}`,
          )}
        >
          <span onClick={() => setIsExpanded(!isExpanded)}>...</span>
          <span
            onClick={() => setIsExpanded(!isExpanded)}
            className={tw("px-1")}
          >
            {brackets.close}
          </span>
        </span>
      )}
    </div>
  );
}

function TruncationNotice({ length, shown, onShowMore }) {
  if (!length || shown >= length) return null;

  return (
    <span
      className={tw(
        `text-gray-500 text-sm ${onShowMore ? "cursor-pointer hover:text-gray-700" : ""}`,
      )}
      onClick={onShowMore}
    >
      ... ({length - shown} more)
    </span>
  );
}

function ArrayPreview({ data, shape }) {
  const isVector = shape && shape.length === 1;
  const isMatrix = shape && shape.length === 2;

  if (isVector && Array.isArray(data)) {
    return (
      <div className={tw("flex flex-wrap gap-1 max-h-32 overflow-y-auto")}>
        {data.slice(0, 50).map((item, i) => (
          <span key={i} className={tw("text-gray-400 text-xs font-mono")}>
            {typeof item === "number" ? item.toFixed(3) : String(item)}
          </span>
        ))}
        {data.length > 50 && (
          <span className={tw("text-gray-500 text-xs")}>...</span>
        )}
      </div>
    );
  }

  if (isMatrix && Array.isArray(data) && Array.isArray(data[0])) {
    return (
      <div className={tw("overflow-x-auto")}>
        <table className={tw("text-xs font-mono")}>
          <tbody>
            {data.slice(0, 8).map((row, i) => (
              <tr key={i}>
                {row.slice(0, 12).map((cell, j) => (
                  <td key={j} className={tw("px-1.5 py-0.5 text-right")}>
                    {typeof cell === "number" ? cell.toFixed(3) : String(cell)}
                  </td>
                ))}
                {row.length > 12 && (
                  <td className={tw("px-1.5 py-0.5 text-gray-500")}>...</td>
                )}
              </tr>
            ))}
            {data.length > 8 && (
              <tr>
                <td
                  colSpan={Math.min(12, data[0]?.length || 0) + 1}
                  className={tw("px-1.5 py-0.5 text-center text-gray-500")}
                >
                  ...
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    );
  }

  // For higher dimensional arrays or non-standard formats
  return (
    <pre
      className={tw(
        "text-xs bg-gray-100 p-2 rounded overflow-x-auto max-h-32 font-mono",
      )}
    >
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

function CompactDictItems({ items, length }) {
  const groupedItems = [];
  let currentGroup = [];

  items.forEach((item) => {
    const isPrimitive =
      item.value?.type_info?.category === "builtin" &&
      !["list", "tuple", "set", "dict"].includes(item.value?.type_info?.type);

    if (!isPrimitive && currentGroup.length > 0) {
      groupedItems.push({ items: currentGroup, inline: true });
      currentGroup = [];
    }

    currentGroup.push(item);

    if (!isPrimitive) {
      groupedItems.push({ items: currentGroup, inline: false });
      currentGroup = [];
    }
  });

  if (currentGroup.length > 0) {
    groupedItems.push({ items: currentGroup, inline: true });
  }

  return (
    <div className={tw("space-y-2")}>
      {groupedItems.map((group, groupIdx) => (
        <div key={groupIdx}>
          {group.inline ? (
            <div className={tw("flex flex-wrap gap-2")}>
              {group.items.map((item, i) => (
                <span key={i}>
                  <InspectValue data={item.key} inline />
                  <span className={tw("mx-1")}>:</span>
                  <InspectValue data={item.value} inline />
                  {i < group.items.length - 1 && (
                    <span className={tw("ml-2 text-gray-400")}>,</span>
                  )}
                </span>
              ))}
            </div>
          ) : (
            <div>
              {group.items.map((item, i) => (
                <div key={i} className={tw("mb-2")}>
                  <InspectValue data={item.key} inline />
                  <span className={tw("mx-1")}>:</span>
                  <InspectValue data={item.value} />
                </div>
              ))}
            </div>
          )}
          {groupIdx < groupedItems.length - 1 && (
            <div className={tw("border-t border-gray-200 my-2")} />
          )}
        </div>
      ))}
      {length > items.length && (
        <TruncationNotice length={length} shown={items.length} />
      )}
    </div>
  );
}

function CompactCollectionItems({ items, length }) {
  const groupedItems = [];
  let currentGroup = [];

  items.forEach((item) => {
    const isPrimitive =
      item?.type_info?.category === "builtin" &&
      !["list", "tuple", "set", "dict"].includes(item?.type_info?.type);

    if (!isPrimitive && currentGroup.length > 0) {
      groupedItems.push({ items: currentGroup, inline: true });
      currentGroup = [];
    }

    currentGroup.push(item);

    if (!isPrimitive) {
      groupedItems.push({ items: currentGroup, inline: false });
      currentGroup = [];
    }
  });

  if (currentGroup.length > 0) {
    groupedItems.push({ items: currentGroup, inline: true });
  }

  return (
    <div className={tw("space-y-2")}>
      {groupedItems.map((group, groupIdx) => (
        <div key={groupIdx}>
          {group.inline ? (
            <div className={tw("flex flex-wrap gap-1")}>
              {group.items.map((item, i) => (
                <React.Fragment key={i}>
                  <InspectValue data={item} inline />
                  {i < group.items.length - 1 && (
                    <span className={tw("text-gray-400")}>,</span>
                  )}
                </React.Fragment>
              ))}
            </div>
          ) : (
            <div>
              {group.items.map((item, i) => (
                <div key={i} className={tw("mb-2")}>
                  <InspectValue data={item} />
                </div>
              ))}
            </div>
          )}
          {groupIdx < groupedItems.length - 1 && (
            <div className={tw("border-t border-gray-200 my-2")} />
          )}
        </div>
      ))}
      {length > items.length && (
        <div className={tw("mt-2")}>
          <TruncationNotice length={length} shown={items.length} />
        </div>
      )}
    </div>
  );
}

function PandasTable({ data, columns, dtypes, shape, truncated }) {
  return (
    <div className={tw("space-y-2")}>
      <div className={tw("text-sm text-gray-600")}>
        Shape: {shape[0]} rows × {shape[1]} columns
      </div>

      <div className={tw("overflow-x-auto")}>
        <table className={tw("text-sm border-collapse w-full")}>
          <thead>
            <tr className={tw("bg-gray-50")}>
              {columns.map((col) => (
                <th
                  key={col}
                  className={tw("px-3 py-2 text-left border font-medium")}
                >
                  <div>{col}</div>
                  <div className={tw("text-xs text-gray-500 font-normal")}>
                    {dtypes[col]}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i} className={tw("hover:bg-gray-50")}>
                {columns.map((col) => (
                  <td key={col} className={tw("px-3 py-2 border")}>
                    <InspectValue
                      data={{
                        type_info: {
                          type: typeof row[col],
                          category: "builtin",
                        },
                        value: row[col],
                      }}
                      inline
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {truncated && (
        <div className={tw("text-sm text-gray-500 italic")}>
          ... and {shape[0] - data.length} more rows
        </div>
      )}
    </div>
  );
}

function InspectValue({ data, inline = false }) {
  // Handle primitives - return them directly

  if (
    typeof data === "boolean" ||
    typeof data === "number" ||
    typeof data === "string"
  ) {
    return <span className={tw(PRIMITIVE_COLORS[typeof data])}>{data}</span>;
  }
  if (!data) {
    return <span className={tw(PRIMITIVE_COLORS["null"])}>null</span>;
  }
  if (!data.type_info) {
    return <span className={tw("text-gray-500 italic")}>Invalid data</span>;
  }

  const {
    type_info,
    value,
    truncated,
    length,
    shape,
    dtype,
    columns,
    dtypes,
    attributes,
  } = data;

  // Handle simple primitives with minimal styling (including numpy scalars)
  if (
    type_info.category === "builtin" ||
    (type_info.category === "numpy" && typeof value === "number" && !shape)
  ) {
    if (type_info.type === "NoneType") {
      return <span className={tw(PRIMITIVE_COLORS.null)}>None</span>;
    }

    if (
      type_info.type === "int" ||
      type_info.type === "float" ||
      type_info.type.startsWith("int") ||
      type_info.type.startsWith("float") ||
      type_info.type.startsWith("uint") ||
      type_info.type === "complex"
    ) {
      return (
        <span
          className={tw(PRIMITIVE_COLORS.number)}
          data-inspect-type={type_info.type}
          data-inspect-category="builtin"
        >
          {String(value)}
        </span>
      );
    }

    if (type_info.type === "bytes") {
      return (
        <div className={tw("space-y-1")}>
          <div
            className={tw(
              "font-mono max-w-[150px] text-sm bg-gray-100 p-2 rounded truncate",
            )}
            data-inspect-type="bytes"
            data-inspect-category="builtin"
            data-inspect-length={length}
          >
            {value}
            {truncated && "..."}
          </div>
        </div>
      );
    }

    // Handle datetime objects with minimal styling
    if (
      type_info.type === "datetime" ||
      type_info.type === "date" ||
      type_info.type === "time"
    ) {
      return (
        <span
          className={tw(`${PRIMITIVE_COLORS.datetime} inline-block`)}
          data-inspect-type={type_info.type}
          data-inspect-category="builtin"
        >
          {value}
        </span>
      );
    }
  }

  // For inline display of collections, show simple tag
  if (
    inline &&
    (type_info.type === "list" ||
      type_info.type === "tuple" ||
      type_info.type === "set" ||
      type_info.type === "dict")
  ) {
    return (
      <span
        className={tw("text-gray-600 text-sm")}
        data-inspect-type={type_info.type}
        data-inspect-category={type_info.category}
        data-inspect-length={length}
      >
        {type_info.type} ({length})
      </span>
    );
  }

  // Handle collections with compact expandable interface
  if (type_info.type === "dict") {
    return (
      <CompactExpandable typeInfo={type_info} defaultExpanded={length <= 5}>
        <CompactDictItems items={value} length={length} />
      </CompactExpandable>
    );
  }

  if (
    type_info.type === "list" ||
    type_info.type === "tuple" ||
    type_info.type === "set"
  ) {
    return (
      <CompactExpandable typeInfo={type_info} defaultExpanded={length <= 5}>
        <CompactCollectionItems items={value} length={length} />
      </CompactExpandable>
    );
  }

  // Handle NumPy and JAX arrays
  if (type_info.category === "numpy" || type_info.category === "jax") {
    const shapeStr = shape ? shape.join(", ") : "";
    const count = shape ? `(${shapeStr})` : "";

    const headerTag = (
      <span className={tw("text-gray-500 text-xs font-mono")}>
        {dtype || "float64"}
        <span className={tw("ml-0.5")}>{count}</span>
      </span>
    );

    return (
      <CompactExpandable
        typeInfo={{ type: "list" }}
        defaultExpanded={false}
        headerTag={headerTag}
      >
        <ArrayPreview data={value} shape={shape} />
      </CompactExpandable>
    );
  }

  // Handle pandas DataFrames
  if (type_info.category === "pandas" && type_info.type === "DataFrame") {
    const count = shape ? shape[0] * shape[1] : undefined;
    return (
      <div className={tw("border rounded-lg overflow-hidden")}>
        <div className={tw(`px-3 py-2 bg-gray-50`)}>
          <span className={tw("text-xs font-mono")}>
            {type_info.type}
            {count && <span className={tw("ml-[2px]")}>({count})</span>}
          </span>
        </div>
        <div className={tw("px-3 py-2 bg-white border-t")}>
          <PandasTable
            data={value}
            columns={columns}
            dtypes={dtypes}
            shape={shape}
            truncated={truncated}
          />
        </div>
      </div>
    );
  }

  // Handle pandas Series
  if (type_info.category === "pandas" && type_info.type === "Series") {
    return (
      <div className={tw("border rounded-lg overflow-hidden")}>
        <div className={tw(`px-3 py-2 bg-gray-50`)}>
          <span className={tw("text-xs font-mono")}>
            {type_info.type}
            {shape && <span className={tw("ml-[2px]")}>({shape[0]})</span>}
          </span>
        </div>
        <div className={tw("px-3 py-2 bg-white border-t")}>
          <ArrayPreview data={value} shape={shape} />
        </div>
      </div>
    );
  }

  // Handle custom objects with attributes
  if (attributes && attributes.length > 0) {
    return (
      <div className={tw("border rounded-lg overflow-hidden")}>
        <div className={tw(`px-3 py-2 bg-gray-50`)}>
          <span className={tw("text-xs font-mono")}>
            {type_info.type}
            <span className={tw("ml-[2px]")}>({attributes.length})</span>
          </span>
        </div>
        <div className={tw("px-3 py-2 bg-white border-t")}>
          <div className={tw("space-y-2")}>
            <div className={tw("text-sm bg-gray-100 p-2 rounded font-mono")}>
              {value}
            </div>
            {attributes.length > 0 && (
              <div>
                <h4 className={tw("font-medium text-sm mb-2")}>Attributes:</h4>
                <CompactDictItems
                  items={attributes}
                  length={attributes.length}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Handle any remaining scalar numeric types that might have been missed
  if (typeof value === "number" && !shape && !attributes) {
    return (
      <span
        className={tw(PRIMITIVE_COLORS.number)}
        data-inspect-type={type_info.type}
        data-inspect-category={type_info.category}
      >
        {String(value)}
      </span>
    );
  }

  // Fallback for other types
  return (
    <div className={tw("space-y-2")}>
      <div
        className={tw("text-sm bg-gray-100 p-2 rounded font-mono break-all")}
        data-inspect-type={type_info.type}
        data-inspect-category={type_info.category}
      >
        {String(value)}
      </div>
      {data.error && (
        <div className={tw("text-sm text-red-600 italic")}>
          Error: {data.error}
        </div>
      )}
    </div>
  );
}

function TypeTooltip({ type, length, visible, x, y }) {
  if (!visible) return null;

  let typeDisplay = type;
  if (type === "float") {
    typeDisplay = "float64";
  } else if (type === "int") {
    typeDisplay = "int64";
  }

  let details = "";
  if (length !== undefined) {
    if (type === "str") {
      details = ` • ${length} chars`;
    } else if (["list", "tuple", "set", "dict"].includes(type)) {
      details = ` • ${length} items`;
    } else if (type === "bytes") {
      details = ` • ${length} bytes`;
    }
  }

  return (
    <div
      className={tw(
        "absolute z-50 px-2 py-1 bg-gray-800 text-white text-xs rounded shadow-lg pointer-events-none whitespace-nowrap",
      )}
      style={{ left: x, top: y - 35 }}
    >
      <span className={tw("font-mono")}>{typeDisplay}</span>
      {details && <span>{details}</span>}
    </div>
  );
}

export function inspect({ data }) {
  const [tooltip, setTooltip] = useState({
    visible: false,
    type: "",
    length: undefined,
    x: 0,
    y: 0,
  });
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleMouseEnter = (e) => {
      const target = e.target.closest("[data-inspect-type]");
      if (target) {
        const rect = target.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        setTooltip({
          visible: true,
          type: target.dataset.inspectType,
          length: target.dataset.inspectLength,
          x: rect.left - containerRect.left + rect.width / 2,
          y: rect.top - containerRect.top,
        });
      }
    };

    const handleMouseLeave = (e) => {
      const target = e.target.closest("[data-inspect-type]");
      if (target) {
        setTooltip((prev) => ({ ...prev, visible: false }));
      }
    };

    container.addEventListener("mouseenter", handleMouseEnter, true);
    container.addEventListener("mouseleave", handleMouseLeave, true);

    return () => {
      container.removeEventListener("mouseenter", handleMouseEnter, true);
      container.removeEventListener("mouseleave", handleMouseLeave, true);
    };
  }, []);

  if (!data) {
    return (
      <div className={tw("text-red-500")}>No data provided to inspect</div>
    );
  }

  return (
    <div ref={containerRef} className={tw("max-w-full relative")}>
      <InspectValue data={data} />
      <TypeTooltip {...tooltip} />
    </div>
  );
}

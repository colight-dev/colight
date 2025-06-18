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

// TypeTag component for expandable containers

function TypeTag({ typeInfo, count, isExpanded, onClick }) {
  return (
    <span className={tw(`inline-block text-xs font-mono`)} onClick={onClick}>
      {isExpanded !== undefined && (
        <span
          className={tw(
            `inline-block transform transition-transform mr-1 ${isExpanded ? "rotate-90" : "rotate-0"}`,
          )}
        >
          ▶
        </span>
      )}
      {typeInfo.type}
      {count && <span className={tw("ml-[2px]")}>({count})</span>}
    </span>
  );
}

function ExpandableContainer({
  typeInfo,
  count,
  children,
  defaultExpanded = false,
}) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className={tw("border rounded-lg overflow-hidden")}>
      <div
        className={tw(`px-3 py-2 bg-gray-50 cursor-pointer hover:bg-gray-100`)}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <TypeTag typeInfo={typeInfo} count={count} isExpanded={isExpanded} />
      </div>
      {isExpanded && (
        <div className={tw("px-3 py-2 bg-white border-t")}>{children}</div>
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

function ArrayPreview({ data, shape, dtype }) {
  const isVector = shape && shape.length === 1;
  const isMatrix = shape && shape.length === 2;

  if (isVector && Array.isArray(data)) {
    return (
      <div className={tw("space-y-2")}>
        <div className={tw("text-sm text-gray-600")}>
          Shape: [{shape.join(", ")}]{dtype && ` • dtype: ${dtype}`}
        </div>
        <div className={tw("flex flex-wrap gap-1 max-h-32 overflow-y-auto")}>
          {data.slice(0, 20).map((item, i) => (
            <span
              key={i}
              className={tw("px-2 py-1 bg-gray-100 rounded text-sm font-mono")}
            >
              {typeof item === "number" ? item.toFixed(3) : String(item)}
            </span>
          ))}
          {data.length > 20 && <span className={tw("text-gray-500")}>...</span>}
        </div>
      </div>
    );
  }

  if (isMatrix && Array.isArray(data) && Array.isArray(data[0])) {
    return (
      <div className={tw("space-y-2")}>
        <div className={tw("text-sm text-gray-600")}>
          Shape: [{shape.join(", ")}]{dtype && ` • dtype: ${dtype}`}
        </div>
        <div className={tw("overflow-x-auto")}>
          <table className={tw("text-sm font-mono border-collapse")}>
            <tbody>
              {data.slice(0, 5).map((row, i) => (
                <tr key={i}>
                  {row.slice(0, 10).map((cell, j) => (
                    <td key={j} className={tw("px-2 py-1 border text-right")}>
                      {typeof cell === "number"
                        ? cell.toFixed(3)
                        : String(cell)}
                    </td>
                  ))}
                  {row.length > 10 && (
                    <td className={tw("px-2 py-1 text-gray-500")}>...</td>
                  )}
                </tr>
              ))}
              {data.length > 5 && (
                <tr>
                  <td
                    colSpan={Math.min(10, data[0]?.length || 0)}
                    className={tw("px-2 py-1 text-center text-gray-500")}
                  >
                    ...
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // For higher dimensional arrays or non-standard formats
  return (
    <div className={tw("space-y-2")}>
      {shape && (
        <div className={tw("text-sm text-gray-600")}>
          Shape: [{shape.join(", ")}]{dtype && ` • dtype: ${dtype}`}
        </div>
      )}
      <pre
        className={tw(
          "text-sm bg-gray-100 p-2 rounded overflow-x-auto max-h-32",
        )}
      >
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}

function KeyValueTable({ items, length }) {
  return (
    <div className={tw("space-y-1")}>
      {items.map((item, i) => (
        <div
          key={i}
          className={tw(
            "flex gap-2 py-1 border-b border-gray-100 last:border-b-0",
          )}
        >
          <div
            className={tw("font-medium text-gray-700 min-w-0 flex-shrink-0")}
          >
            <InspectValue data={item.key} inline />:
          </div>
          <div className={tw("min-w-0 flex-1")}>
            <InspectValue data={item.value} inline />
          </div>
        </div>
      ))}
      <TruncationNotice length={length} shown={items.length} />
    </div>
  );
}

function CollectionItems({ items, length }) {
  return (
    <div className={tw("space-y-1")}>
      <div className={tw("flex flex-wrap items-center gap-1")}>
        {items.map((item, i) => (
          <React.Fragment key={i}>
            <span className={tw("inline-block")}>
              <InspectValue data={item} inline />
            </span>
            {i < items.length - 1 && (
              <span className={tw("text-gray-400")}>,</span>
            )}
          </React.Fragment>
        ))}
        {length > items.length && (
          <>
            <span className={tw("text-gray-400")}>,</span>
            <TruncationNotice length={length} shown={items.length} />
          </>
        )}
      </div>
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

  // Handle collections with expandable interface
  if (type_info.type === "dict") {
    return (
      <ExpandableContainer
        typeInfo={type_info}
        count={length}
        defaultExpanded={length <= 10}
      >
        <KeyValueTable items={value} length={length} />
      </ExpandableContainer>
    );
  }

  if (
    type_info.type === "list" ||
    type_info.type === "tuple" ||
    type_info.type === "set"
  ) {
    return (
      <ExpandableContainer
        typeInfo={type_info}
        count={length}
        defaultExpanded={length <= 10}
      >
        <CollectionItems items={value} length={length} />
      </ExpandableContainer>
    );
  }

  // Handle NumPy and JAX arrays
  if (type_info.category === "numpy" || type_info.category === "jax") {
    const count = shape ? shape.reduce((a, b) => a * b, 1) : undefined;
    return (
      <ExpandableContainer
        typeInfo={type_info}
        count={count}
        defaultExpanded={true}
      >
        <ArrayPreview data={value} shape={shape} dtype={dtype} />
      </ExpandableContainer>
    );
  }

  // Handle pandas DataFrames
  if (type_info.category === "pandas" && type_info.type === "DataFrame") {
    const count = shape ? shape[0] * shape[1] : undefined;
    return (
      <ExpandableContainer
        typeInfo={type_info}
        count={count}
        defaultExpanded={true}
      >
        <PandasTable
          data={value}
          columns={columns}
          dtypes={dtypes}
          shape={shape}
          truncated={truncated}
        />
      </ExpandableContainer>
    );
  }

  // Handle pandas Series
  if (type_info.category === "pandas" && type_info.type === "Series") {
    return (
      <ExpandableContainer
        typeInfo={type_info}
        count={shape ? shape[0] : undefined}
        defaultExpanded={shape[0] <= 20}
      >
        <ArrayPreview data={value} shape={shape} dtype={dtype} />
      </ExpandableContainer>
    );
  }

  // Handle custom objects with attributes
  if (attributes && attributes.length > 0) {
    return (
      <ExpandableContainer
        typeInfo={type_info}
        count={attributes.length}
        defaultExpanded={false}
      >
        <div className={tw("space-y-2")}>
          <div className={tw("text-sm bg-gray-100 p-2 rounded font-mono")}>
            {value}
          </div>
          {attributes.length > 0 && (
            <div>
              <h4 className={tw("font-medium text-sm mb-2")}>Attributes:</h4>
              <KeyValueTable items={attributes} length={attributes.length} />
            </div>
          )}
        </div>
      </ExpandableContainer>
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

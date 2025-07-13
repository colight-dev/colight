import { useState } from "react";
import { tw } from "../../../colight/src/js/utils";

// Individual node in the directory tree
const DirectoryNode = ({
  node,
  onSelectFile,
  onNavigateToDirectory,
  level = 0,
}) => {
  const isDirectory = node.type === "directory";
  const hasChildren = isDirectory && node.children && node.children.length > 0;
  const [expanded, setExpanded] = useState(false);

  // Visual collapsing: if this directory has only one child and it's also a directory,
  // show them collapsed together
  let displayName = node.name;
  let collapsedChild = null;

  if (
    isDirectory &&
    node.children?.length === 1 &&
    node.children[0].type === "directory"
  ) {
    // Check if the single child also has a single directory child, and so on
    let current = node;
    const parts = [node.name];

    while (
      current.children?.length === 1 &&
      current.children[0].type === "directory"
    ) {
      current = current.children[0];
      parts.push(current.name);
    }

    if (parts.length > 1) {
      displayName = parts.join("/");
      collapsedChild = current;
    }
  }

  const handleClick = () => {
    if (isDirectory) {
      // // If we have a collapsed child, navigate to its path instead
      // const targetNode = collapsedChild || node;
      // const targetPath = targetNode.path || targetNode.name;
      // onNavigateToDirectory(targetPath);

      // Toggle expansion separately
      if (hasChildren) {
        setExpanded(!expanded);
      }
    } else {
      // For files, use the full path
      onSelectFile(node.path || node.name);
    }
  };

  return (
    <div style={{ paddingLeft: `${level * 20}px` }}>
      <div
        className={tw(
          "flex items-center py-1.5 px-2 cursor-pointer transition-all duration-200 hover:bg-gray-50 rounded",
        )}
        onClick={handleClick}
      >
        {isDirectory && (
          <span
            className={tw(
              `mr-2 text-xs transition-transform duration-200 ${
                expanded ? "" : "-rotate-90"
              }`,
            )}
            style={{ width: "12px", display: "inline-block" }}
          >
            {hasChildren ? "▼" : ""}
          </span>
        )}
        {!isDirectory && (
          <span
            className={tw("mr-2 text-gray-300")}
            style={{ width: "12px", display: "inline-block" }}
          >
            •
          </span>
        )}
        <span
          className={tw(
            `${
              isDirectory
                ? "font-medium text-gray-700"
                : "text-gray-600 hover:text-gray-900"
            } ${displayName.includes("/") ? "text-blue-600" : ""}`,
          )}
        >
          {displayName}
        </span>
      </div>
      {isDirectory && hasChildren && expanded && (
        <div>
          {/* If we collapsed directories, show the deepest directory's children */}
          {(collapsedChild?.children || node.children).map((child, idx) => (
            <DirectoryNode
              key={child.name + "-" + idx}
              node={child}
              onSelectFile={onSelectFile}
              onNavigateToDirectory={onNavigateToDirectory}
              level={level + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// Find a specific directory in the tree
const findDirectory = (tree, targetPath) => {
  if (!targetPath || targetPath === "/" || targetPath === "") {
    return tree;
  }

  // Normalize the target path - remove trailing slashes
  const normalizedTarget = targetPath.endsWith("/")
    ? targetPath.slice(0, -1)
    : targetPath;

  const traverse = (node) => {
    // Check this node
    const nodePath = (node.path || node.name || "").replace(/\/$/, "");
    if (nodePath === normalizedTarget) {
      return node;
    }

    // Check children
    if (node.children) {
      for (const child of node.children) {
        const result = traverse(child);
        if (result) return result;
      }
    }

    return null;
  };

  return traverse(tree);
};

export const DirectoryBrowser = ({
  directoryPath,
  tree,
  onSelectFile,
  onNavigateToDirectory,
}) => {
  // Find the subtree for the given directory path
  const subtree = findDirectory(tree, directoryPath);

  if (!subtree) {
    return (
      <div className={tw("text-gray-500")}>
        Directory not found: {directoryPath}
      </div>
    );
  }

  // Show children if this is a directory with children
  if (subtree.children && subtree.children.length > 0) {
    return (
      <div>
        {subtree.children.map((child, idx) => (
          <DirectoryNode
            key={child.name + "-" + idx}
            node={child}
            onSelectFile={onSelectFile}
            onNavigateToDirectory={onNavigateToDirectory}
          />
        ))}
      </div>
    );
  }

  // Empty directory
  if (subtree.type === "directory") {
    return (
      <div className={tw("text-gray-500 text-center py-8")}>
        This directory is empty
      </div>
    );
  }

  // Single file (shouldn't happen for directory browser but handle it)
  return (
    <DirectoryNode
      node={subtree}
      onSelectFile={onSelectFile}
      onNavigateToDirectory={onNavigateToDirectory}
    />
  );
};

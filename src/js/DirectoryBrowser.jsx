import { useState, } from "react";
import { tw } from "../../packages/colight/src/colight/js/utils";

// Individual node in the directory tree
const DirectoryNode = ({ node, onSelectFile, level = 0, defaultExpanded = false }) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const isDirectory = node.type === "directory";
  const hasChildren = isDirectory && node.children && node.children.length > 0;
  
  const handleClick = () => {
    if (isDirectory) {
      if (hasChildren) {
        setExpanded(!expanded);
      }
    } else {
      // Remove .py extension for loading
      const cleanPath = node.path.replace(/\.py$/, "");
      onSelectFile(cleanPath);
    }
  };
  
  return (
    <div style={{ paddingLeft: `${level * 20}px` }}>
      <div
        className={tw(
          "flex items-center py-1.5 px-2 cursor-pointer transition-all duration-200 hover:bg-gray-50 rounded"
        )}
        onClick={handleClick}
      >
        {isDirectory && (
          <span 
            className={tw(`mr-2 text-xs transition-transform duration-200 ${
              expanded ? "" : "-rotate-90"
            }`)}
            style={{ width: "12px", display: "inline-block" }}
          >
            {hasChildren ? "▼" : ""}
          </span>
        )}
        {!isDirectory && (
          <span className={tw("mr-2 text-gray-300")} style={{ width: "12px", display: "inline-block" }}>
            •
          </span>
        )}
        
        <span className={tw(`flex-1 ${isDirectory ? "text-gray-700 font-medium" : "text-gray-600"} text-sm`)}>
          {node.name.replace(/\.py$/, "")}
        </span>
      </div>
      
      {isDirectory && hasChildren && expanded && (
        <div>
          {node.children.map((child) => (
            <DirectoryNode
              key={child.path}
              node={child}
              onSelectFile={onSelectFile}
              level={level + 1}
              defaultExpanded={false}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// Main directory browser component
export const DirectoryBrowser = ({ directoryPath, onSelectFile, onClose, tree }) => {
  // Get the subtree for the current directory
  let displayTree = tree;
  
  if (tree && directoryPath && directoryPath !== "/") {
    const parts = directoryPath.split("/").filter(Boolean);
    let current = tree;
    
    for (const part of parts) {
      if (current.type === "directory" && current.children) {
        const child = current.children.find(c => c.name === part);
        if (child) {
          current = child;
        } else {
          break;
        }
      }
    }
    
    displayTree = current;
  }
  
  if (!displayTree) {
    return null;
  }
  
  return (
    <div className={tw("max-w-3xl mx-auto p-4")}>
      {displayTree.type === "directory" && displayTree.children ? (
        displayTree.children.map((child) => (
          <DirectoryNode
            key={child.path}
            node={child}
            onSelectFile={onSelectFile}
            defaultExpanded={false}
          />
        ))
      ) : (
        <DirectoryNode
          node={displayTree}
          onSelectFile={onSelectFile}
          defaultExpanded={true}
        />
      )}
    </div>
  );
};
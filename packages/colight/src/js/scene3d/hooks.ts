/**
 * React hooks for scene3d.
 */

import React from "react";

/**
 * Hook to track container width with optional threshold for debouncing updates.
 */
export function useContainerWidth(
  threshold: number = 10,
): [React.RefObject<HTMLDivElement>, number] {
  const containerRef = React.useRef<HTMLDivElement>(null!);
  const [containerWidth, setContainerWidth] = React.useState<number>(0);
  const lastWidthRef = React.useRef<number>(0);

  React.useEffect(() => {
    if (!containerRef.current) return;

    const handleWidth = (width: number) => {
      const diff = Math.abs(width - lastWidthRef.current);
      if (diff >= threshold) {
        lastWidthRef.current = width;
        setContainerWidth(width);
      }
    };

    // Initial measurement
    handleWidth(containerRef.current.offsetWidth);

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        handleWidth(entry.contentRect.width);
      }
    });

    observer.observe(containerRef.current);

    return () => observer.disconnect();
  }, [threshold]);

  return [containerRef, containerWidth];
}

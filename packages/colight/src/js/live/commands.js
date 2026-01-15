export const getCommands = ({ pragmaOverrides, setPragmaOverrides }) => {
  return [
    {
      type: "toggle",
      title: `${pragmaOverrides.hideStatements ? "Show" : "Hide"} Statements`,
      subtitle: `${pragmaOverrides.hideStatements ? "○" : "●"} Toggle statement visibility`,
      searchTerms: ["statements", "hide", "show", "toggle"],
      action: () =>
        setPragmaOverrides((prev) => ({
          ...prev,
          hideStatements: !prev.hideStatements,
        })),
    },
    {
      type: "toggle",
      title: `${pragmaOverrides.hideCode ? "Show" : "Hide"} Code`,
      subtitle: `${pragmaOverrides.hideCode ? "○" : "●"} Toggle code visibility`,
      searchTerms: ["code", "hide", "show", "toggle"],
      action: () =>
        setPragmaOverrides((prev) => ({ ...prev, hideCode: !prev.hideCode })),
    },
    {
      type: "toggle",
      title: `${pragmaOverrides.hideProse ? "Show" : "Hide"} Prose`,
      subtitle: `${pragmaOverrides.hideProse ? "○" : "●"} Toggle prose visibility`,
      searchTerms: ["prose", "text", "markdown", "hide", "show", "toggle"],
      action: () =>
        setPragmaOverrides((prev) => ({
          ...prev,
          hideProse: !prev.hideProse,
        })),
    },
    {
      type: "toggle",
      title: `${pragmaOverrides.hideVisuals ? "Show" : "Hide"} Visuals`,
      subtitle: `${pragmaOverrides.hideVisuals ? "○" : "●"} Toggle visual outputs`,
      searchTerms: ["visuals", "visual", "output", "hide", "show", "toggle"],
      action: () =>
        setPragmaOverrides((prev) => ({
          ...prev,
          hideVisuals: !prev.hideVisuals,
        })),
    },
  ];
};

- fix the parser/generator such that forms are an _ordered_ list of elements (prose, expressions, statements). Each element is shown/hidden according to the applicable pragma. However, only the last element of a form is relevant in considering whether we show a visual or not (ie. a visual is only shown when hide-visuals is not true and the last element of a form is an expression).


- modify index_generator.py to prepare JSON which is rendered by live.jsx. 




UX/SPA
- workflowy-style outliner for files-being-watched

- update forms granularly; keep scroll position

LATER
- pin/split windows and visuals 


- implement a python MCP server for evaluating python, with optional support for image generation (of visuals). Look at the clojure tools as an example.

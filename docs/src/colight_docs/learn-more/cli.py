# %% [markdown]
# # CLI Tools
#
# Colight ships a command-line interface as part of the `colight-prose` package.
# It powers the documentation build workflow by executing notebook-style `.py`
# files (like the one you are reading) and exporting them to Markdown.
#
# %% [markdown]
# ## Where the CLI is documented
#
# The canonical reference lives in `packages/colight-prose/README.md`. This page
# surfaces the highlights of that README and keeps it connected to the rest of
# the docs site so you can find it through navigation.
#
# %% [markdown]
# ## Core commands
#
# | Command | When to use it | Notes |
# | --- | --- | --- |
# | `colight-prose build <SOURCE> --output <DEST>` | One-off conversion of `.py` docs into Markdown or HTML. | Useful in CI and for testing whether a single file compiles. |
# | `colight-prose watch <SOURCE> --output <DEST>` | Incrementally rebuilds docs as you edit files. | Pass `--no-dev-server` to skip starting the live preview server. |
# | `colight-prose live <SOURCE>` | Starts the live editing server backed by the watch pipeline. | Great for pairing Markdown edits with interactive previews. |
#
# SOURCE may be a single file or a directory tree. DEST can be a file or folder
# depending on the command.
#
# %% [markdown]
# ## Common options
#
# - `--output PATH` (or `-o PATH`): Choose where built markdown/HTML files are
#   written.
# - `--no-dev-server`: Use with `watch` when you only need file rebuilds.
# - `--format markdown|html`: Override the default export format set in
#   `mkdocs.yml`.
# - `--quiet / --verbose`: Control progress output when running in CI.
#
# Options supplied on the CLI sit between file-level defaults and the form-level
# pragmas described below, so you can override a default without touching the
# source file.
#
# %% [markdown]
# ## Doc file format
#
# Doc sources are executable Python files where comment blocks become Markdown
# and executable statements emit Colight visuals. The minimal loop looks like:
#
# ```python
# # %% [markdown]
# # # My Page
# # A line of markdown lives inside a Python comment.
#
# import colight.plot as Plot
# my_data = [[0, 1], [1, 2], [2, 3]]
# Plot.line(my_data)
# ```
#
# Comments that start with `# %% [markdown]` introduce a Markdown cell. Plain
# comments at the top of a file are also lifted into prose, which is why many
# docs read naturally when opened as `.py` files.
#
# %% [markdown]
# ## Pragma directives
#
# Pragmas let you control what gets rendered without editing CLI arguments.
#
# ### File-level pragmas
#
# Start-of-file directives such as `# | hide-all-code` or `# | hide-all-visuals`
# apply to the entire document and establish defaults. They are useful for
# notebooks where you want the prose to lead.
#
# ### Form-level pragmas
#
# Place `# %% hide-code`, `# | show-visuals`, or `#| colight: hide-prose` right
# before a code cell to override the defaults for that specific block. Form-level
# directives always win over file-level settings.
#
# ### Supported toggles
#
# - `hide-code` / `show-code`
# - `hide-statements` / `show-statements`
# - `hide-visuals` / `show-visuals`
# - `hide-prose` / `show-prose`
# - `format-markdown` / `format-html`
#
# Use whichever syntax is most convenient—double-percent, pipe, or legacy
# `#| colight:`—they all target the same underlying flags.
#
# ### Precedence
#
# `form-level` > `CLI options` > `file-level` > defaults. Explicit `show-*`
# directives also override any matching `hide-*` directive.
#

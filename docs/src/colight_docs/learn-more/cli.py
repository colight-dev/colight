# %% [markdown]
# # CLI Tools
#
# Colight ships a command-line interface as part of the `colight-publish` package,
# exposed to end users as `colight publish`. It powers the documentation build
# workflow by executing notebook-style `.py` files (like the one you are reading)
# and exporting them to Markdown, HTML, or the full interactive site.
#
# %% [markdown]
# ## Where the CLI is documented
#
# The canonical reference lives in `packages/colight-publish/README.md`. This page
# surfaces the highlights of that README and keeps it connected to the rest of
# the docs site so you can find it through navigation.
#
# %% [markdown]
# ## Core command
#
# Everything now flows through a single entry point:
#
# ```bash
# colight publish <SOURCE> --format (md|html|site) [--watch] [--output DEST]
# ```
#
# | Format | Watch? | What you get | Default output |
# | --- | --- | --- | --- |
# | `md` | optional | Markdown files only | `build/` |
# | `html` | optional | Standalone HTML + `.colight` assets | `build/` (or `.colight_cache/` when watching with the dev server) |
# | `site` | optional | The interactive explorer UI | `site-build/` (static) |
#
# SOURCE may be a single file or a directory tree. DEST can be a file or folder
# depending on the command. For `site --watch` the CLI runs the LiveServer
# directly and ignores `--output`.
#
# %% [markdown]
# ## Common options
#
# - `--output PATH` (or `-o PATH`): Choose where artifacts land (ignored for `site --watch`).
# - `--watch`: Regenerate on change. Starts the appropriate dev server for `html`
#   and `site`; Markdown mode simply rewrites files.
# - `--include/--ignore`: Glob filters for picking files out of a docs tree.
# - `--verbose`: Print extra progress information.
# - `--colight-output-path`, `--colight-embed-path`, `--inline-threshold`: Tuning
#   knobs for HTML and Markdown output.
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

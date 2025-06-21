"""Constants for colight-site."""

# Default size threshold (in bytes) for inlining .colight files as script tags
# Files smaller than this will be embedded directly, larger files use external references
DEFAULT_INLINE_THRESHOLD = 50000

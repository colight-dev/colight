
def parse_pragma_arg(pragma):
  """Parse pragma tags from comma-separated string"""
  if not pragma:
    return set()
  if isinstance(pragma, set):
    return pragma
  return {tag.strip() for tag in pragma.split(",") if tag.strip()}
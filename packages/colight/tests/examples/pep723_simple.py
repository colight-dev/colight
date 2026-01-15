# /// script
# dependencies = [
#   "pandas",
#   "tabulate",
# ]
# ///

# Simple PEP 723 Example
# This demonstrates that dependencies are automatically installed

import pandas as pd
from tabulate import tabulate

# Create a simple dataframe
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "London", "Tokyo"],
}

df = pd.DataFrame(data)

# Display using tabulate
print(tabulate(df, headers="keys", tablefmt="grid"))

# Return the dataframe for visualization
df

# # Visual Update Test
# This tests if visuals update when code changes

# %%
import colight.plot as Plot

from .visual_update_dep import x

x + 1


# Change this value to test visual updates.
value = 12  # <-- EDIT THIS NUMBER

# Create a simple plot using Observable Plot


(
    Plot.dot(
        [
            [1, value],
            [2, value + 10],
            [3, value + 20],
            [4, value + 30],
            [5, value + 40],
        ],
        x=0,
        y=1,
    )
    + Plot.title(f"Test Plot with value={value}")
    + {"height": 30}
)

# When you change the `value` variable above, the plot should update to show the new data.

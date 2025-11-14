# # Scroll Test Document
# This document is designed to test the scroll behavior when blocks update.

# First code block


print("Block 1 - Static content")
x = 2

# ## Section 2
# Some prose between blocks to create vertical space.
# This helps us see if the scroll position is maintained.

# Second code block - modify this to test
print("Block 2 - CHANGE ME to test scroll!")
y = 2
f"x + y = {x + y}"

# ## Section 3
# More content to ensure we have enough height for scrolling.
# Lorem ipsum dolor sit amet, consectetur adipiscing elit.

# Third code block
import datetime

f"Block 3 - Current time: {datetime.datetime.now()}"
z = x + y

# ## Section 4
# Even more content to create vertical space.
# - Item 1
# - Item 2
# - Item 3

# Fourth code block
print("Block 4 - More static content")
result = x * y * z
f"Result: {result}"

# ## Final Section
# This is the end of the document.
# When you modify Block 2 above, the page should:
# 1. NOT jump to the top
# 2. Show pending states for dirty blocks
# 3. Update blocks in document order
# 4. Maintain your scroll position
# 5. If only one block changed, smoothly scroll to it

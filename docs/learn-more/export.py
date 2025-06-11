# ## Exporting HTML, Screenshots & Videos
#
# Colight provides methods to save visualizations as HTML files, images, or videos.
#
# ### HTML Files
#
# Save a plot as a standalone HTML file:
#
# ```python
# plot.save_html("my_plot.html")
# ```
#
# ### Static Images
#
# Save a plot as a static image:
#
# ```python
# plot.save_image("my_plot.png", width=800)
# ```
#
# The `width` and `height` parameters are optional and specified in pixels. Height will be measured if not supplied.
#
# ### Image Sequences and Videos
#
# Save multiple states of a plot as separate images:
#
# ```python
# plot.save_images(
#     state_updates=[
#         {"counter": 1},
#         {"counter": 10},
#         {"counter": 100}
#     ],
#     output_dir="./frames",
#     filename_base="count"
# )
# ```
#
# Save state transitions as a video (requires ffmpeg):
#
# ```python
# # Save as MP4 video
# plot.save_video(
#     state_updates=[{"counter": i} for i in range(30)],
#     filename="animation.mp4",
#     fps=24
# )
#
# # Or save as animated GIF
# plot.save_video(
#     state_updates=[{"counter": i} for i in range(30)],
#     filename="animation.gif",
#     fps=24
# )
# ```
#
# Note: Video generation requires ffmpeg to be installed on your system.

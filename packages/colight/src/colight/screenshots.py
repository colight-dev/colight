"""
Screenshot utilities for Colight plots using a StudioContext which inherits from ChromeContext
"""

import base64
import json
import time
import subprocess  # Added import for subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import colight.widget as widget
import colight.format as format

import colight.env as env
from colight.chrome_devtools import ChromeContext, format_bytes
from colight.util import read_file


class StudioContext(ChromeContext):
    """
    StudioContext extends ChromeContext with Colight-specific methods.
    It encapsulates behavior such as loading the Colight environment, rendering plots, and updating state.
    """

    def __init__(
        self,
        plot=None,
        data=None,
        buffers=None,
        reuse=True,
        keep_alive: float = 1.0,
        window_vars=None,
        **kwargs,
    ):
        """
        Initialize StudioContext with optional plot

        Args:
            plot: Optional plot to load on initialization
            data: Pre-serialized plot data (optional)
            buffers: Pre-serialized buffers (optional)
            window_vars: Dict of variables to set on window object before loading content
            **kwargs: Additional arguments passed to ChromeContext
        """
        self._plot = plot
        self._data = data
        self._buffers = buffers
        super().__init__(
            reuse=reuse, keep_alive=keep_alive, window_vars=window_vars, **kwargs
        )

    def __enter__(self):
        context = super().__enter__()
        if self._plot is not None:
            self.load_plot(self._plot, data=self._data, buffers=self._buffers)
        return context

    def load_studio_html(self):
        # Check if Colight environment is already loaded
        if not self.evaluate("typeof window.colight === 'object'"):
            if self.debug:
                print("[screenshots.py] Loading Colight HTML")

            files = {}
            # Handle script content based on whether env.WIDGET_PATH is a CDN URL or local file
            if isinstance(env.WIDGET_PATH, str):  # CDN URL
                if self.debug:
                    print(f"[screenshots.py] Using CDN script from: {env.WIDGET_PATH}")
                script_tag = f'<script type="module" src="{env.WIDGET_PATH}"></script>'
            else:  # Local file
                if self.debug:
                    print(
                        f"[screenshots.py] Loading local script from: {env.WIDGET_PATH}"
                    )
                script_tag = '<script type="module" src="studio.js"></script>'
                files["studio.js"] = read_file(env.WIDGET_PATH)

            # CSS is now embedded in the JS bundle - no separate styling needed
            style_tag = ""

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset=\"UTF-8\">
                <title>Colight</title>
                {style_tag}
                {script_tag}
            </head>
            <body>
                <div id=\"studio\"></div>
            </body>
            </html>
            """
            self.load_html(html, files=files)
        elif self.debug:
            print("Colight already loaded, skipping initialization")

    def load_plot(self, plot=None, data=None, buffers=None, measure=True):
        """
        Loads the plot in the Colight environment.

        Args:
            plot: The plot to load (required if data/buffers not provided)
            data: Pre-serialized plot data (optional)
            buffers: Pre-serialized buffers (optional)
            measure: Whether to measure container size
        """
        if self.debug:
            print("[StudioContext] Loading plot into Colight")

        self.load_studio_html()

        # Use provided data/buffers if available, otherwise serialize
        if data is not None and buffers is not None:
            colight_data = format.create_bytes(data, buffers)
        else:
            data, buffers = widget.to_json_with_state(plot, buffers=[])
            colight_data = format.create_bytes(data, buffers)
        colight_filename = f"plot_{self.id}.colight"
        self.server.add_served_file(colight_filename, colight_data)
        colight_url = f"http://localhost:{self.server_port}/{colight_filename}"

        if self.debug:
            print(
                f"[StudioContext] Serving .colight file: {colight_url} ({format_bytes(len(colight_data))})"
            )
            print(f"[StudioContext] Contains {len(buffers)} buffers")

        render_js = f"""
         (async () => {{
           console.log('[StudioContext] Loading .colight file for ID: {self.id}');
           try {{
             const colightData = await window.colight.loadColightFile('{colight_url}');
             await window.colight.render('studio', colightData, '{self.id}');
             await window.colight.whenReady('{self.id}');
           }} catch (error) {{
             console.error('[StudioContext] Failed to load .colight file:', error);
           }}
         }})()
         """
        self.evaluate(render_js, await_promise=True)

        if measure:
            self.measure_size()

    def measure_size(self):
        """
        Measures container size and adjusts context dimensions accordingly.
        """
        dimensions = self.evaluate("""
            (function() {
                const container = document.querySelector('.colight-container');
                if (!container) return null;
                const rect = container.getBoundingClientRect();
                return { width: Math.ceil(rect.width), height: Math.ceil(rect.height) };
            })()
        """)
        if self.debug:
            print(f"[StudioContext] Measured container dimensions: {dimensions}")
        if dimensions is not None:
            self.set_size(dimensions["width"], dimensions["height"])

    def update_state(self, state_updates):
        """
        Sends state updates to Colight. Expects state_updates to be a list.
        """
        if self.debug:
            print("[StudioContext] Updating state")
        if not isinstance(state_updates, list):
            raise AssertionError("state_updates must be a list")
        collected_state = widget.CollectedState()
        state_data = widget.to_json(state_updates, collected_state=collected_state)
        buffers = collected_state.buffers

        # Convert buffers to base64 for passing to JavaScript (for state updates, not initial load)
        encoded_buffers = [
            base64.b64encode(buffer).decode("utf-8") for buffer in buffers
        ]

        update_js = f"""
        (async function() {{
            try {{
                const updates = {json.dumps(state_data)}
                const buffers = {json.dumps(encoded_buffers)}.map(b64 =>
                    Uint8Array.from(atob(b64), c => c.charCodeAt(0))
                );
                const result = window.colight.instances['{self.id}'].updateWithBuffers(updates, buffers);
                await window.colight.whenReady('{self.id}');
                return result;
            }} catch (e) {{
                console.error('State update failed:', e);
                return 'error: ' + e.message;
            }}
        }})()
        """
        return self.evaluate(update_js, await_promise=True)

    def capture_image_sequence(
        self, state_updates: List[Dict], format: str = "png", quality: int = 90
    ) -> List[bytes]:
        """
        Capture a sequence of images after applying each state update.

        Args:
            state_updates: List of state updates to apply before each capture
            format: Image format ("png" or "webp")
            quality: Image quality for WebP format (0-100, ignored for PNG)

        Returns:
            List of image bytes in the specified format
        """
        bytes_list = []
        for state_update in state_updates:
            self.update_state([state_update])
            image_bytes = self.capture_bytes(format=format, quality=quality)
            bytes_list.append(image_bytes)
        return bytes_list

    def save_image(
        self,
        output_path: Optional[Union[str, Path]] = None,
        state_update: Optional[Dict] = None,
        quality: int = 90,
    ) -> Union[Path, bytes]:
        """
        Save an image of the current plot state.

        Args:
            output_path: Optional path to save the image; if not provided, returns PNG bytes
            state_update: Optional state update to apply before capturing
            quality: Image quality for WebP format (0-100, ignored for PNG)

        Returns:
            Path to saved image if output_path provided, otherwise PNG bytes
        """
        if state_update:
            self.update_state([state_update])

        # Infer format from file extension
        if output_path:
            out_path = Path(output_path)
            ext = out_path.suffix.lower()
            if ext == ".webp":
                format = "webp"
            elif ext == ".png":
                format = "png"
            else:
                # Default to PNG for unknown extensions
                format = "png"

            image_bytes = self.capture_bytes(format=format, quality=quality)
            out_path.parent.mkdir(exist_ok=True, parents=True)
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            if self.debug:
                print(f"[StudioContext] Image saved to: {out_path}")
            return out_path
        else:
            # Default to PNG when returning bytes
            format = "png"
            image_bytes = self.capture_bytes(format=format, quality=quality)
            return image_bytes

    def save_image_sequence(
        self,
        state_updates: List[Dict],
        output_dir: Union[str, Path],
        filenames: Optional[List[str]] = None,
        filename_base: Optional[str] = "screenshot",
        quality: int = 90,
    ) -> List[Path]:
        """
        Save a sequence of images after applying each state update.

        Args:
            state_updates: List of state updates to apply before each capture
            output_dir: Directory where images will be saved
            filenames: Optional list of filenames for each image
            filename_base: Base name for generating filenames if not provided
            quality: Image quality for WebP format (0-100, ignored for PNG)

        Returns:
            List of paths to saved images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        if filenames:
            if len(filenames) != len(state_updates):
                raise ValueError(
                    f"Number of filenames ({len(filenames)}) must match number of state updates ({len(state_updates)})"
                )
        else:
            filenames = [f"{filename_base}_{i}.png" for i in range(len(state_updates))]

        output_paths = [output_dir / name for name in filenames]

        # Infer format from first filename (assume all files use same format)
        first_ext = output_paths[0].suffix.lower() if output_paths else ".png"
        if first_ext == ".webp":
            format = "webp"
        else:
            format = "png"

        saved_paths = []

        image_bytes_list = self.capture_image_sequence(
            state_updates, format=format, quality=quality
        )
        for i, (image_bytes, out_path) in enumerate(
            zip(image_bytes_list, output_paths)
        ):
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            saved_paths.append(out_path)
            if self.debug:
                print(
                    f"[StudioContext] Saved image {i+1}/{len(state_updates)} to: {out_path}"
                )

        return saved_paths

    def save_pdf(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> Union[Path, bytes]:
        """
        Save a PDF of the current plot state.

        Args:
            output_path: Optional path to save the PDF; if not provided, returns PDF bytes

        Returns:
            Path to saved PDF if output_path provided, otherwise PDF bytes
        """

        # Trigger WebGPU canvas capture for 3D content before PDF generation
        self.evaluate(
            f"window.colight.beforeScreenCapture('{self.id}');", await_promise=True
        )

        # Capture the PDF content (including static images of 3D canvases)
        pdf_bytes = self.capture_pdf()

        # Cleanup and restore interactive 3D content
        self.evaluate(
            f"window.colight.afterScreenCapture('{self.id}');", await_promise=True
        )

        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(exist_ok=True, parents=True)
            with open(out_path, "wb") as f:
                f.write(pdf_bytes)
            if self.debug:
                print(f"[StudioContext] PDF saved to: {out_path}")
            return out_path
        return pdf_bytes

    def capture_bytes(self, format: str = "png", quality: int = 90):
        """Capture image bytes in specified format."""
        self.evaluate(
            f"window.colight.beforeScreenCapture('{self.id}');", await_promise=True
        )
        bytes = self.capture_image(format=format, quality=quality)
        self.evaluate(
            f"window.colight.afterScreenCapture('{self.id}');", await_promise=True
        )
        return bytes

    def capture_video(
        self,
        state_updates: List[Dict],
        filename: Union[str, Path],
        fps: int = 24,
    ) -> Path:
        """
        Capture a series of states from a plot as a video.
        The video is generated without saving intermediate images to disk by piping PNG frames
        directly to ffmpeg.

        Args:
            state_updates: List of state update dictionaries to apply sequentially
            filename: Path where the resulting video will be saved
            fps: Frame rate (frames per second) for the video

        Returns:
            Path to the saved video file
        """
        filename = Path(filename)
        if self.debug:
            print(f"[StudioContext] Recording video with {len(state_updates)} frames")

        start_time = time.time()
        filename.parent.mkdir(exist_ok=True, parents=True)

        # Detect file extension
        ext = filename.suffix.lower()

        if ext == ".gif":
            ffmpeg_cmd = (
                f"ffmpeg {'-v error' if not self.debug else ''} -y "
                f"-f image2pipe -vcodec png -framerate {fps} -i - "
                # The filter below: (1) splits the pipeline into two streams;
                #                   (2) generates a palette from one stream;
                #                   (3) applies that palette to the other stream;
                #                   (4) loops infinitely (0) in the final GIF.
                f"-vf \"split [a][b];[b]palettegen=stats_mode=diff[p];[a][p]paletteuse=new=1\" "
                f"-c:v gif -loop 0 \"{filename}\""
            )
        else:
            # Generate MP4 video with libx264 - using yuv420p for QuickTime compatibility
            ffmpeg_cmd = (
                f"ffmpeg {'-v error' if not self.debug else ''} -y "
                f"-f image2pipe -vcodec png -framerate {fps} -i - "
                f"-an -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow \"{filename}\""
            )

        if self.debug:
            print(f"[StudioContext] Running ffmpeg command: {ffmpeg_cmd}")

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, shell=True)

        try:
            for i, state_update in enumerate(state_updates):
                result = self.update_state([state_update])
                if self.debug:
                    print(f"[StudioContext] State update {i} result: {result}")

                frame_bytes = self.capture_bytes()

                if proc.stdin:
                    proc.stdin.write(frame_bytes)
                    if self.debug:
                        print(f"[StudioContext] Captured frame {i}")

            if proc.stdin:
                proc.stdin.close()
            proc.wait()

            elapsed_time = time.time() - start_time
            actual_fps = len(state_updates) / elapsed_time
            if self.debug:
                print(
                    f"[StudioContext] Video generation took {elapsed_time:.2f} seconds (~{actual_fps:.1f} fps)"
                )

            return filename
        except Exception as e:
            # Clean up process on error
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            raise e


def save_image(
    plot,
    output_path: Optional[Union[str, Path]] = None,
    state_update: Optional[Dict] = None,
    data: Optional[Any] = None,
    buffers: Optional[List[bytes]] = None,
    width: int = 400,
    height: Optional[int] = None,
    scale: float = 1.0,
    quality: int = 90,
    debug: bool = False,
    reuse: bool = True,
    keep_alive: float = 1.0,
    window_vars: Optional[Dict[str, Any]] = None,
) -> Union[Path, bytes]:
    """
    Render the plot and capture an image.

    Args:
        plot: The Colight plot widget
        output_path: Optional path to save the image; if not provided, returns PNG bytes
        state_update: Optional state update to apply before capture
        data: Pre-serialized plot data (optional)
        buffers: Pre-serialized buffers (optional)
        width: Width of the browser window
        height: Optional height of the browser window
        scale: Device scale factor
        quality: Image quality for WebP format (0-100, ignored for PNG)
        debug: Whether to print debug information
        window_vars: Dict of variables to set on window object before loading content

    Returns:
        Path to saved image if output_path is provided, otherwise PNG bytes
    """
    with StudioContext(
        plot=plot,
        data=data,
        buffers=buffers,
        width=width,
        height=height,
        scale=scale,
        debug=debug,
        reuse=reuse,
        keep_alive=keep_alive,
        window_vars=window_vars,
    ) as studio:
        return studio.save_image(output_path, state_update, quality)


def save_images(
    plot,
    state_updates: List[Dict],
    output_dir: Union[str, Path] = "./scratch/screenshots",
    filenames: Optional[List[str]] = None,
    filename_base: Optional[str] = "screenshot",
    data: Optional[Any] = None,
    buffers: Optional[List[bytes]] = None,
    width: int = 800,
    height: Optional[int] = None,
    scale: float = 1.0,
    quality: int = 90,
    debug: bool = False,
    reuse: bool = True,
    keep_alive: float = 1.0,
    window_vars: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """
    Capture a sequence of images with state updates.

    Args:
        plot: The Colight plot widget
        state_updates: List of state update dictionaries to apply sequentially
        output_dir: Directory where images will be saved
        filenames: Optional list of filenames for each image; if not provided, filenames will be auto-generated
        filename_base: Base name for generating filenames
        data: Pre-serialized plot data (optional)
        buffers: Pre-serialized buffers (optional)
        width: Width of the browser window
        height: Optional height of the browser window
        scale: Device scale factor
        quality: Image quality for WebP format (0-100, ignored for PNG)
        debug: Whether to print debug information
        window_vars: Dict of variables to set on window object before loading content

    Returns:
        List of paths to the saved images
    """
    with StudioContext(
        plot=plot,
        data=data,
        buffers=buffers,
        width=width,
        height=height,
        scale=scale,
        debug=debug,
        reuse=reuse,
        keep_alive=keep_alive,
        window_vars=window_vars,
    ) as studio:
        return studio.save_image_sequence(
            state_updates, output_dir, filenames, filename_base, quality
        )


def save_pdf(
    plot,
    output_path: Optional[Union[str, Path]] = None,
    data: Optional[Any] = None,
    buffers: Optional[List[bytes]] = None,
    width: int = 400,
    height: Optional[int] = None,
    scale: float = 1.0,
    debug: bool = False,
    reuse: bool = True,
    keep_alive: float = 1.0,
    window_vars: Optional[Dict[str, Any]] = None,
) -> Union[Path, bytes]:
    """
    Render the plot and capture a PDF of the page.

    Args:
        plot: The Colight plot widget
        output_path: Optional path to save the PDF; if not provided, returns PDF bytes
        data: Pre-serialized plot data (optional)
        buffers: Pre-serialized buffers (optional)
        width: Width of the browser window
        height: Optional height of the browser window
        scale: Device scale factor
        debug: Whether to print debug information
        window_vars: Dict of variables to set on window object before loading content

    Returns:
        Path to saved PDF if output_path is provided, otherwise PDF bytes
    """
    with StudioContext(
        plot=plot,
        data=data,
        buffers=buffers,
        width=width,
        height=height,
        scale=scale,
        debug=debug,
        reuse=reuse,
        keep_alive=keep_alive,
        window_vars=window_vars,
    ) as studio:
        return studio.save_pdf(output_path)


def save_video(
    plot,
    filename: Union[str, Path],
    state_updates: Optional[List[Dict]] = None,
    fps: Optional[int] = None,
    data: Optional[Any] = None,
    buffers: Optional[List[bytes | bytearray | memoryview]] = None,
    width: int = 400,
    height: Optional[int] = None,
    scale: float = 1.0,
    debug: bool = False,
    reuse: bool = True,
    keep_alive: float = 1.0,
    window_vars: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a video of a plot animation.

    If the plot contains a slider with fps specified, it will automatically be used
    for animation. Otherwise, state_updates must be provided.

    Args:
        plot: The Colight plot widget
        filename: Path where the resulting video will be saved (.mp4 or .gif)
        state_updates: List of state update dictionaries to apply sequentially.
                      Optional if the plot has a slider with fps specified.
        fps: Frame rate for the video. If None and the plot has an animated slider,
             uses the slider's fps. Otherwise defaults to 24.
        data: Pre-serialized plot data (optional, for performance)
        buffers: Pre-serialized buffers (optional, for performance)
        width: Width of the browser window (default: 400)
        height: Height of the browser window (optional)
        scale: Device scale factor (default: 1.0)
        debug: Whether to print debug information
        reuse: Whether to reuse existing browser instance
        keep_alive: Time to keep browser alive after completion
        window_vars: Dict of variables to set on window object before loading content

    Returns:
        Path to the saved video file

    Examples:
        # Automatic animation from slider
        plot = Plot.dot(data, x="time", y="value") + Plot.Slider("time", range=100, fps=30)
        save_video(plot, "animation.mp4")

        # Manual state updates
        save_video(plot, "custom.mp4", state_updates=[{"x": i} for i in range(10)])

    Raises:
        ValueError: If neither state_updates nor an animated slider is available
    """
    # If state_updates not provided, try to use animateBy metadata
    if state_updates is None:
        # Ensure we have serialized data
        if data is None or buffers is None:
            data, buffers = widget.to_json_with_state(plot, buffers=[])

        animateBy = data.get("animateBy") if data else None
        if not animateBy:
            raise ValueError(
                "No state_updates provided and no animated slider found in plot"
            )
        # Check if there are multiple animated sliders
        if len(animateBy) > 1:
            raise ValueError(
                f"Multiple animated sliders found ({len(animateBy)}). "
                "Please provide explicit state_updates when using multiple sliders."
            )

        # Use the single animated slider
        animateBy = animateBy[0]

        # Generate state updates from metadata
        (from_, to_) = animateBy["range"]

        state_updates = [
            {animateBy["key"]: i} for i in range(from_, to_ + 1, animateBy["step"])
        ]

        # Use fps from metadata if not provided
        if fps is None:
            fps = animateBy["fps"]

    with StudioContext(
        plot=plot,
        data=data,
        buffers=buffers,
        width=width,
        height=height,
        scale=scale,
        debug=debug,
        reuse=reuse,
        keep_alive=keep_alive,
        window_vars=window_vars,
    ) as studio:
        out = studio.capture_video(state_updates, filename, fps or 24)
        return out

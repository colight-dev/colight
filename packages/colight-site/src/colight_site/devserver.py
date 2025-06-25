from pathlib import Path
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.middleware.shared_data import SharedDataMiddleware
from livereload import Server
import nest_asyncio

# Allow nested event loops in Jupyter
nest_asyncio.apply()

roots = {
    "/dist": Path("dist").resolve(),  # main site
    "/": Path("test-artifacts/colight-site").resolve(),  # extra files
}


class HtmlFallbackMiddleware:
    def __init__(self, app, roots):
        self.app = app
        self.roots = roots

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")

        # Check if it's a naked path (no extension)
        if not Path(path).suffix and not path.endswith("/"):
            for mount, root in self.roots.items():
                if path.startswith(mount):
                    relative_path = path[len(mount) :].lstrip("/")
                    html_file = root / f"{relative_path}.html"

                    if html_file.exists():
                        redirect_url = f"{path}.html"
                        start_response("302 Found", [("Location", redirect_url)])
                        return [b""]

        # Check for root path and redirect to index.html
        elif path == "/" or path == "":
            for mount, root in self.roots.items():
                if mount == "/":
                    index_file = root / "index.html"
                    if index_file.exists():
                        start_response("302 Found", [("Location", "/index.html")])
                        return [b""]

        return self.app(environ, start_response)


# build a simple dispatcher that serves each prefix
def not_found_app(_environ, start_response):
    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not Found"]


app = SharedDataMiddleware(
    not_found_app, {mount: str(root) for mount, root in roots.items()}
)
app = HtmlFallbackMiddleware(app, roots)  # Add HTML fallback
app = DispatcherMiddleware(app, {})  # required by livereload

if __name__ == "__main__":
    server = Server(app)
    for root in roots.values():
        server.watch(str(root / "**/*"))  # watch everything below each dir
    server.serve(root=None, host="127.0.0.1", port=5500, open_url_delay=1)

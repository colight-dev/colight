from colight.layout import js


def canvas_mark(user_canvas_fn):
    return js(
        """(_indexes, scales, _values, dim, _ctx) => {
    const devicePixelRatio = window.devicePixelRatio || 1;

    /* ---- build the canvas-in-SVG wrapper -------------------------------- */
    const svgNS = "http://www.w3.org/2000/svg";
    const fo    = document.createElementNS(svgNS, "foreignObject");
    fo.setAttribute("width",  dim.width);
    fo.setAttribute("height", dim.height);

    const canvas = document.createElement("canvas");
    
    // Set actual canvas size (high-res)
    canvas.width  = dim.width * devicePixelRatio;
    canvas.height = dim.height * devicePixelRatio;
    
    // Set CSS size (display size)
    canvas.style.width = dim.width + 'px';
    canvas.style.height = dim.height + 'px';
    fo.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    ctx.scale(devicePixelRatio, devicePixelRatio);
                   
    user_canvas_fn(ctx, scales)
    
    return fo;
}""",
        user_canvas_fn=user_canvas_fn,
    )

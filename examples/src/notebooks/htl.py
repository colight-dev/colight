import colight.plot as plot

svg_defs = plot.js("""
() => {
    const svg = colight.api.htl.svg`<defs>
       <pattern id="myFill" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(-45)">
         <line x1="0" x2="10" stroke="currentColor" />
       </pattern>
     </defs>`

    return svg               

}
""")

plot.new(
    svg_defs,
    plot.rectY(
        [{"animal": "sheep", "count": 1}, {"animal": "cow", "count": 3}],
        {"x": "animal", "y": "count", "fill": "url(#myFill)", "stroke": "currentColor"},
    ),
)

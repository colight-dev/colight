# | hide-all-code

import os
import glob
import colight.plot as Plot

py_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
py_files = [f for f in py_files if os.path.basename(f) != "index.py"]
file_names = [os.path.basename(f) for f in py_files]

Plot.html(
    [
        "div.p-6",
        ["h1.text-2xl.font-bold.mb-6", "Colight Examples Directory"],
        ["p.mb-4", "Click on any example file to view it:"],
        [
            "ul.list-disc.pl-6",
            *[
                [
                    "li.mb-2",
                    [
                        "a.text-blue-600.hover:underline",
                        {"href": f"{name.replace('.py', '')}"},
                        name,
                    ],
                ]
                for name in sorted(file_names)
            ],
        ],
    ]
)

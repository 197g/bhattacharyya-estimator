// Render the whole docs folder as a typst document.
#import "@preview/cmarker:0.1.3"
#import "@preview/mitex:0.2.4": mitex

#let pages = (
	"reading-material.md",
)

#for page in pages [
    #cmarker.render(read(page), math: mitex, h1-level: 2, raw-typst: false)
]

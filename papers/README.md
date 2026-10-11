# The method behind AVA

AVA builds on pseudo-Boolean polynomials (PBP) by Tendai M. Chikake and Boris Goldengorin.
Our implementation in `ava/` is independent; the papers below describe the underlying method.

## Main paper

**T. M. Chikake, B. I. Goldengorin, P. M. Pardalos.**
*Pseudo-Boolean Polynomial Method for Interpretable Dimensionality Reduction: A Paradigm Shift from Abstract to Meaningful Feature Extraction.*
Computer Optics, 2025, 49(6), pp. 1191–1201. DOI: [10.18287/COJ1815](https://doi.org/10.18287/COJ1815)

> A general-purpose, training-free framework for dimensionality reduction and clustering based on per-sample
> pseudo-Boolean polynomials (PBP).

What AVA takes from it: the construction of a polynomial from a small matrix (sort each column, turn every step into a
monomial of the rows below it, merge equal monomials), the worked example
`11 + 11y3 + 3y4 + 2y1y3 + 4y2y3 + 4y1y4 + 12y1y2y3 + 6y1y2y4` (reproduced in `tests/test_pbp.py`), and the idea that
the polynomial's monomial structure is an interpretable description of the data.

What AVA adds (built during the hackathon): applying it to 8×8 image patches in two orientations, ε-ties, contrast
normalisation, tolerant and multi-reference comparison, alignment, glare masking, defect zones, the single-photo surface
mode and the benchmarks in `eval/`.

## Related publications

- T. M. Chikake, B. Goldengorin. *Image edge detection using pseudo-Boolean polynomials.* ICMV 2023, Proc. SPIE 13072.
  DOI: [10.1117/12.3023452](https://doi.org/10.1117/12.3023452)
- T. M. Chikake, B. Goldengorin, A. Samosyuk. *Pseudo-Boolean polynomials approach to edge detection and image
  segmentation.* Springer, 2023. DOI: [10.1007/978-3-031-31654-8_5](https://doi.org/10.1007/978-3-031-31654-8_5)
- T. M. Chikake, B. Goldengorin, P. M. Pardalos. Chapter in *Trends in Biomathematics* (BIOMAT 2025), Springer.
  DOI: [10.1007/978-3-031-97461-8_3](https://doi.org/10.1007/978-3-031-97461-8_3)
- The authors' Python library: [`tmc-pbp`](https://pypi.org/project/tmc-pbp/) (MIT).

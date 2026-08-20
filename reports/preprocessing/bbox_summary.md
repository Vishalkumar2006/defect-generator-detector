# KSDD2 defect bounding-box analysis

**Status: PASS**

All coordinates are pixel coordinates with inclusive maxima. Connected components use 8-connectivity.

## Defect counts

- all: 356
- development_train: 209
- validation: 37
- test: 110

## Geometry quantiles (all defective images)

| Measure | Min | Median | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| bbox_width | 7 | 65 | 109.25 | 158.5 | 186.25 | 230.45 | 232 |
| bbox_height | 5 | 68.5 | 113 | 203 | 262.25 | 438.3 | 637 |
| mask_pixels | 23 | 2273 | 5105 | 8852.5 | 13251 | 30685.2 | 43869 |
| mask_fraction | 0.000170854 | 0.0155614 | 0.0346189 | 0.0617168 | 0.090981 | 0.209892 | 0.30158 |

## Border contact

- All defective images: 169
- Development-training defective images: 90

## Complete-defect fit in development training

Reflection padding may supply image background near an image edge, but cannot make an oversized defect count as fitting.

| Patch width × height | Context | Fits | Total | Percentage |
|---|---:|---:|---:|---:|
| 256x256 | 0 px | 196 | 209 | 93.78% |
| 256x256 | 16 px | 186 | 209 | 89.00% |
| 256x256 | 32 px | 179 | 209 | 85.65% |
| 256x384 | 0 px | 204 | 209 | 97.61% |
| 256x384 | 16 px | 196 | 209 | 93.78% |
| 256x384 | 32 px | 191 | 209 | 91.39% |
| 256x512 | 0 px | 207 | 209 | 99.04% |
| 256x512 | 16 px | 200 | 209 | 95.69% |
| 256x512 | 32 px | 194 | 209 | 92.82% |

## Recommendation

No candidate meets the rule: smallest candidate containing at least 95% of development-training defects with 32 pixels of context.

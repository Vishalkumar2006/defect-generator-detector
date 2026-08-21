# Experiment configurations

`final_real_baseline.json` is retained unchanged as the historical failed FP16
experiment configuration. It failed numerical-stability requirements during epoch
4 validation and must not be resumed or used as the final reference. See
`docs/failed-fp16-baseline.md` for its disposition and preserved evidence.

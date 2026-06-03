# Model/My preprocessing QC

This `Model/My` folder is a preprocessing-only reset for visual QC.

These images are for visual inspection of whether nuclei and bud-neck-related fluorescence are visible enough before building prompts.

Current scope:
- project GFP and mCherry stacks so they align with `Trans`
- enhance fluorescence contrast
- save per-frame `Trans`, GFP-only, mCherry-only, merged RGB, and merged RGB enhanced PNGs
- build contact sheets for quick review

Not included in this step:
- no boxes
- no centroids
- no pair/single labels
- no SAM

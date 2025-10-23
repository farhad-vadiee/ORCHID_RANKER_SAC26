# Data Availability

The datasets used in the paper are not redistributed as part of this
anonymous artifact to comply with their original licenses. To reproduce
the experiments:

1. Download the raw EdNet and OULAD releases from their official sources.
2. Place the raw archives under `data/` following the structure
   described in `configs/`.
3. Run the preprocessing entrypoints
   (`orchid-preprocess-ednet` and `orchid-preprocess-oulad`) to generate
   the processed CSV bundles consumed by the experiment drivers.

All preprocessing scripts and experiment configurations needed for
reproduction are included in this repository.

# DATASET

## Heart beat intervals
The RR models expect three files:`tr-rr.csv`, `sl-rr.csv` and `ts-rr.csv`. Respectively: training data, validation data and test data.
Every file is expected to have 4*N+2 columns with N being the maximum length of an rr sequence.
For each i in \[0, N-1\] there are four columns `pre_l{i}`, `pre_g{i}`, `post_l{i}`, `post_g{i}`
- pre/post describe whether the sample has been taken before of after a breathing exercise
- l/g refers to the z-normalization, **l**ocal to the rr streak or **g**lobal to all the dataset
- {i} is the position of the rr in the streak

Two additional columns are expected `length_pre` and `length_post`, the actual number of rr in the sample.
## HRV features
Feature based model expect three files: `tr.csv`, `sl.csv` and `ts.csv`. Respectively: training data, validation data and test data. The following columns are expected, all z-normalized:
- `post_RMSSD`
- `post_SVI`
- `pre_SDNN`
- `post_averageNN`
- `post_LF`
- `post_SD1`
- `post_SD2`
- `pre_pNN50`
- `pre_SD2`
- `pre_SD1`
- `post_HF`
- `pre_averageNN`
- `pre_SVI`
- `post_SDSD`
- `pre_LF`
- `pre_HF`
- `pre_SDSD`
- `post_SDNN`
- `pre_RMSSD`
- `post_pNN50`
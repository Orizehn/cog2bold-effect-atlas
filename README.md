# Cog2BOLD Effect Atlas

An interactive viewer for a trained **Cog2BOLD** model, which predicts a person's task-fMRI
activation map from their cognitive test scores. The app asks the model *how the predicted map
changes when the scores change*, finds the blobs of activation and deactivation, and reports
them the way a neuroimaging paper does: peak MNI coordinates, Harvard-Oxford and Yeo-7 labels,
and cluster size.

**Read the maps as properties of the model, not of the brain.** Each effect map is what the
trained predictor would change. It is only as meaningful as the model's validated
out-of-sample signal, which the app shows in its Overview tab. The user guide
(`docs/effect_atlas_user_guide.pdf`, also downloadable from the app's sidebar) explains every
view.

## What is included

One trained model, in `bundles/`:
- **Data:** ds000030, paired-associate memory retrieval task (*pamret*), contrast
  `correct_gt_control`.
- **Predictor:** voxelwise ridge regression (identity basis, α = 100), fitted to deviations from
  the mean map.
- **Inputs:** 4 out-of-scanner cognitive scores:
  - `cvlt_totcor`: CVLT total correct over learning trials 1–5;
  - `cvlt_ldf`: CVLT long-delay free recall;
  - `mr_totalraw`: WAIS matrix reasoning;
  - `ssp_totalraw`: WMS spatial span.

The bundle holds the fitted model, the brain mask, held-out metrics, and the training subjects'
cognitive scores and target maps (the bootstrap needs them). Subject IDs are replaced by
anonymous numbers.

## Run it locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On first use, the app downloads the Harvard-Oxford and Schaefer atlases (a few MB) through
nilearn.

## Layout

```
streamlit_app.py      entry point: switches on public mode, then runs apps/effect_atlas.py
apps/                 the Streamlit app
cog2bold/             the Cog2BOLD library (model loading, effect maps, clusters, atlases)
bundles/              self-contained trained-model bundle(s)
docs/                 user guide (PDF)
```

## Data and atlases

- fMRI and behavioural data: UCLA Consortium for Neuropsychiatric Phenomics LA5c study,
  OpenNeuro **ds000030**, released under CC0. Poldrack, R. A. et al. (2016). *A phenome-wide
  examination of neural and cognitive function.* Scientific Data 3, 160110.
- Harvard-Oxford cortical and subcortical probabilistic atlases (FSL), fetched at runtime through
  nilearn.
- Schaefer, A. et al. (2018). *Local-global parcellation of the human cerebral cortex from
  intrinsic functional connectivity MRI.* Cerebral Cortex 28, 3095–3114. 200 parcels, 7 networks
  (Yeo, B. T. T. et al., 2011).

## License

Code: MIT (see `LICENSE`). Data: CC0, as released by OpenNeuro ds000030.

# Datasets

All benchmarks are read by `sgpo/data/benchmarks.py` (`load_split(name, ntrain, nval, ntest, tail_test)`; append
`@sub=k` to a name for the grid-resolution sweeps). Downloaded files live under `datasets/raw/` and generated ones under
`datasets/gen/`; set `LOGOS_RAW` and `LOGOS_GEN` to use other locations. Notebooks `notebooks/data_<name>.ipynb` describe
the data (they read the local files).

## Downloaded (official releases)

`bash scripts/data/download_official.sh [zenodo|drive|all]` fetches:

| Loader name | Source | Location |
|---|---|---|
| `burgers` | FNO authors' `burgers_data_R10.mat` (Li et al. 2021) | `raw/fno/burgers/` |
| `ns_forcing` | CaltechDATA record `fp3ds-kej20` (de Hoop et al. 2022) | `raw/caltech/` |
| `ns1024` | neuraloperator Zenodo record 12825163, `nsforcing_1024.tgz` | unpack to `raw/zenodo/ns1024/` |
| `darcy_std` | FNO Darcy files `piececonst_r421_N1024_smooth1.mat` (Hugging Face mirror) and `..._smooth2.mat` (FNO Google Drive, `Darcy_421.zip`) | `raw/hf_mutransfer/`, unzip to `raw/fno/darcy421/` |
| `airfoil` | Geo-FNO author folders (Google Drive), NACA airfoil | `raw/geofno/airfoil/naca/` |

`co2_sg` uses the U-FNO CO2 gas-saturation data of Wen et al. 2022 (https://github.com/gegewen/ufno): place
`sg_{train,val,test}_{a,u}.pt` in `raw/ufno/`.

## Generated

| Loader name | Command |
|---|---|
| `darcy_notch` | `python scripts/data/gen_darcy_notch.py --n 4000 --out datasets/gen/darcy_notch` |
| `advection_paper` | `python scripts/data/gen_advection.py --n 3000 --out datasets/gen/advection_ic2_200` |

The MATLAB scripts in `data/burger/` are the original Burgers generator (GRF initial conditions).

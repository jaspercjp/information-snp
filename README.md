# Information Theoretic Quantification of Uncertainty in Ensemble Climate Models
Companion notebooks and scripts written to produce results in the paper.

The data used are summarized with the following table from the original thesis:
![image](https://github.com/user-attachments/assets/427d072b-902b-41ce-8964-0e8c75def23c)

The data utilized was not included in this repository, but they can be downloaded and processed with `notebooks/download_data.ipynb` and `notebooks/create_ensemble_dataset.ipynb` along with the the data documents provided in the `data` folder.

When extrapolating datasets, I have chosen to fill unrecorded data with NaN. The calculations are then done either by ignoring them or replacing them with zeros. 

- `notebooks/gaussian_arrays.ipynb` contains code used to study the properties of the information theory predictability measures on randomly generated Gaussian samples. 
- `notebooks/jugaad.ipynb` investigates different questions regarding how the information theory measures change with respect to the size of the ensemble (using the "tiling" jugaad)
- `notebooks/calc_RPC.ipynb` contains the main calculations of the RPC using both variance-based and information-based measures. 

The thesis code is in `scripts/thesis-old/` (the notebooks import it from there), and the thesis figures are in `figures/thesis-old/`:
- `scripts/thesis-old/DataInventory.py` is a utility script used to keep track of all data files downloaded using `notebooks/download_data.ipynb`.
- `scripts/thesis-old/InfoTheoryMetrics.py` contains functions that calculate entropy and mutual information.
- `scripts/thesis-old/info_rpc_parallel.py` is a parallel drop-in for `InfoTheoryMetrics.info_RPC` with bitwise-identical output.
- `scripts/thesis-old/gaussian_array_metrics.py` regenerates `figures/thesis-old/from-notebooks/gaussian-array.png` (the metrics against the correlation r on Gaussian arrays, from `notebooks/gaussian_arrays.ipynb`); run it inside an allocation, with `ml system texlive` for the original's LaTeX fonts.

## Binning uncertainty of the jugaad lambda

`figures/jugaad-fd-optbins/` (see its `README.md` and `BOOTSTRAP.md`) is reproduced end to end by `scripts/`:
- `scripts/data-acquisition/`: raw downloads to the per-case input arrays (`data/cases/`); see its `README.md`.
- `scripts/{jugaad,data,run,boot,tables}.py`: the analysis (499 lines), `scripts/plots.py`: the figures.
- `bash scripts/run_all.sh` submits the whole pipeline to Slurm; `scripts/README.md` has the details. 


# ECE 3715 Mini-Project 3: Random Vectors and the Multivariate Gaussian

## Team members and contributions

| Team member | Assignment |
|---|---|
| Trevor Cheng | Steps 1–3, Bonus A, <code>sample_mean_cov</code>, and <code>is_valid_covariance</code> |
| Andrew Ore | Steps 4–6, Bonus B, <code>gaussian_pdf</code>, and <code>whitening_matrix</code> |
| George Li | Steps 7–8, <code>lmmse_weights</code>, final notebook integration, README, and clean-kernel testing |

## Files

```
mp3/
  mp3.ipynb    the notebook, runs top to bottom (Steps 1–8, then Bonus A and Bonus B)
  mp3lib.py    the five required functions, imported by the notebook
  iris.csv     Fisher's Iris measurements used in Bonus B
  README.md    this file
```

## Run the notebook

Keep mp3.ipynb, mp3lib.py, and iris.csv in the same folder. Open mp3.ipynb in Jupyter and choose **Restart Kernel and Run All Cells**. The notebook uses NumPy, SciPy, and Matplotlib. Bonus B reads Fisher's Iris measurements from the included CSV, which matches the [scikit-learn Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset).

## Seed

```
SEED = 836285155   # drawn 2026-11-18, do not change
```

The seed was drawn once and is hard coded in the notebook's setup cell. That cell creates one generator per step and per bonus section with `rng.spawn(1)[0]`. No other cell calls `default_rng`.

## Runtime

Final clean-kernel run (Restart Kernel and Run All Cells): **8.61 s**, as printed by the notebook's last cell. Including kernel startup, the run took 10.25 s of wall time. The run finished with no errors on Python 3.13.18 with NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8.

## Integration and testing notes

- Steps 7 and 8 were inserted after Step 6, so the eight steps run in order, followed by the two bonus sections.
- After integration, every code cell in Steps 1–6 and both bonus sections produced exactly the same printed output as the Steps 1–6 checkpoint, because each step draws from its own spawned generator.
- Every number quoted in the Step 7 and Step 8 prose appears in that run's printed output.
- Every "check yourself" value for Steps 7 and 8 is reproduced. The Step 8 weights and MMSEs at c = 0, 0.8, 1.0, and 1.4 match the manual to the six decimals it prints.

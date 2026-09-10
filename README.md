<div align="center">

# ArborML

**A Python-based GUI application for tree-based machine learning prediction of material properties**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: GPL](https://img.shields.io/badge/License-GPL-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)]()
[![Version](https://img.shields.io/badge/Version-1.0-brightgreen)]()
[![Languages](https://img.shields.io/badge/UI%20languages-7-orange)]()

</div>

---

## Overview

Machine learning is now routine in materials science, but using it still usually means writing code. **ArborML** (from *arbor*, Latin for tree, and *ML*) closes that gap: it is a cross-platform desktop application that provides a complete, code-free workflow for **tree-based machine learning on tabular experimental data**.

From a standard spreadsheet, ArborML takes you through data exploration, model training, model interpretation, on-demand prediction, and automated PDF reporting ‚All all from a single vertical sidebar.

Unlike general-purpose platforms, ArborML is deliberately narrow in scope. It focuses on **tabular regression with tree-based and linear models**, so the standard sequence of a materials-science ML study becomes the default path rather than something you have to assemble yourself.

## Table of contents

- [Key features](#key-features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Workflow in detail](#workflow-in-detail)
- [Supported data formats](#supported-data-formats)
- [Available models](#available-models)
- [Worked example: soft magnetic composites](#worked-example-soft-magnetic-composites)
- [Interface languages](#interface-languages)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Key features

| | |
|---|---|
| - **Multi-format import** | `.xlsx`, `.xls`, `.csv`, `.txt`, `.dat` , plus a dedicated button for `.json` |
| - **Data exploration** | Per-variable histograms and Pearson correlation heatmaps in one click |
| - **Four tree-based models** | Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| - **Linear regression baseline** | Check whether your variables are linearly dependent before going further |
| - **Editable hyperparameters** | Sensible defaults suggested for every model, all of them adjustable |
| - **Model interpretation** | True vs. predicted plots, R2, RMSE, MSE, MAE, feature importance, SHAP values |
| - **On demand prediction** | Move an input, read the new predicted response immediately ‚no batch file needed |
| - **Model export** | Save a trained model to a self-contained `.arborml` file and reload it later |
| - **PDF report** | Figures, data summary, hyperparameters, metrics, timestamp and all session predictions in one document |
| - **Seven UI languages** | English, Spanish, French, German, Chinese, Japanese, Korean |
| - **Dark / light mode** | Switchable at any time |

## Installation

### Requirements

- **Python** 3.9 or later
- **OS**: Windows 10/11, macOS 12+, or Linux (tested on Ubuntu 26.04.1 LTS)
- **CPU**: x86-64 or ARM64
- **RAM**: 8 GB recommended
- **GPU**: not required

### Install from source

```bash
git clone https://github.com/arborml/arborml.git
cd arborml
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Then install the dependencies and launch:

```bash
pip install -r requirements.txt
python arborml.py
```

### Dependencies

```
customtkinter
pillow
pandas
numpy
openpyxl
scikit-learn
joblib
xgboost
shap
matplotlib
seaborn
reportlab
```

`tkinter` ships with most Python distributions. On Debian/Ubuntu it may need to be installed separately:

```bash
sudo apt install python3-tk
```

## Quick start

1. **Launch** ArborML.
2. Click **Load a file** and select your dataset. The interface displays the file name, the number of rows and columns, and the column names read from the first line (these can be renamed).
3. In **Explore**, click **Histograms** and **Correlation heatmap** to inspect the data.
4. In **Modeling**, pick a model, review the hyperparameters, and train.
5. Read the true-vs-predicted plot, the metrics, the feature importances and the SHAP summary.
6. Click **Simulate results**, set your input values, and read the prediction.
7. Click **Generate report** to export everything to PDF.

## Workflow in detail

### 1. Import

The dataset is expected as a standard tabular file: one row per sample, one column per variable, variable names on the first line. After loading, ArborML reports the dimensions of the dataset and lists the detected columns, which you can rename directly in the interface.

### 2. Explore

- **Histograms** ‚A distribution of each individual variable, useful for spotting gaps, imbalance, or discretised parameters.
- **Correlation heatmap** ‚A Pearson correlation coefficients between all variables, on a -1 to 1 scale.

### 3. Train

Select the target variable and the model. Default hyperparameters are proposed for each estimator and can be edited before training. All estimators are wrapped behind a common interface, so switching between a decision tree and XGBoost changes nothing in your workflow.

After training, ArborML displays:

- the **true vs. predicted** scatter plot on the held-out subset,
- the **performance metrics** (R2, RMSE, MSE, MAE),
- the **feature importance** bar chart,
- the **SHAP** summary, showing how each input variable pushes the prediction.

Trained models can be exported to a `.arborml` file for reuse.

### 4. Predict on demand

This is the part that makes ArborML an exploratory tool rather than a batch predictor. The **Simulate results** window exposes one field per input variable. Change a value, and the predicted response updates immediately. There is no need to prepare a file of candidate conditions in advance.

Every prediction made during the session is buffered in memory and carried into the final report.

Always keep your inputs inside the range covered by the training dataset. Tree-based models do not extrapolate !

### 5. Report

The PDF report assembles, in one self-contained document:

- date and time of generation,
- data summary,
- all figures produced during the session,
- model configuration and hyperparameters,
- performance metrics,
- every interactive prediction made.

This is intended for archiving, sharing with colleagues, and reuse when writing a paper or preparing slides. It also addresses a common source of irreproducibility: data file, hyperparameters and timestamps are usually specified in scripts but omitted from the outputs.

## Supported data formats

| Format | Extension | Import |
|---|---|---|
| Excel | `.xlsx`, `.xls` | Load a file |
| Comma-separated | `.csv` | Load a file |
| Text / data | `.txt`, `.dat` | Load a file |
| JSON | `.json` | Dedicated button |
| Saved ArborML model | `.arborml` | Model loader |

## Available models

| Model | Library | Typical use |
|---|---|---|
| Decision Tree | scikit-learn | Fast, fully interpretable baseline |
| Random Forest | scikit-learn | Robust general-purpose ensemble |
| Gradient Boosting | scikit-learn | Strong accuracy on small tabular datasets |
| XGBoost | xgboost | Regularised boosting, efficient on larger tables |
| Linear Regression | scikit-learn | Reference model to test linear dependence |

> The current version supports **regression on tabular data of moderate size**. Classification is not supported yet.

## Worked example: soft magnetic composites

The repository ships with a worked example based on the openly available dataset of Milyutin, Bures *et al.* Soft magnetic composites (SMCs) were produced from an Fe-3wt% MgO powder mixture by dry mixing, cold pressing and sintering, with five processing parameters varied:

| Parameter | Range |
|---|---|
| MgO particle size | 10 to 1000 nm |
| Compaction pressure | 600 to 1250 MPa |
| Sintering temperature | 350 to 900 °C |
| Sintering time | 0 to 60 min |
| Sintering atmosphere | label encoded: 1 = air, 2 = no annealing, 3 = nitrogen |

This gives **282 samples** with unique parameter combinations. The **resonance frequency** is used as the target variable, and only the six relevant columns are kept.

### Results with Gradient Boosting

| Metric | Value |
|---|---|
| R2 | 0.9939 |
| RMSE | 58.12 kHz |
| MSE | 3378 kHz2 |
| MAE | 31.3 kHz |

These values reproduce those obtained previously with a non-GUI, script-based analysis of the same dataset (R2 = 0.994, RMSE = 58.12 kHz, MAE = 31.304 kHz), confirming that moving the analytical core into a graphical interface preserves the original results. As expected, **sintering time and temperature** dominate the feature importance ranking.

### On-demand prediction

For a composite with 1000 nm MgO particles, pressed at 800 MPa and annealed in air at 400°C for 50 min, the gradient boosting model predicts a resonance frequency of **92.8 kHz** ‚consistent with the low-frequency regime expected at that temperature and annealing time.

**Data sources**

- Milyutin V., Bures R. et al., *The first experimentally obtained data set on processing parameters and properties of soft magnetic composites*, Figshare (2024). https://doi.org/10.6084/m9.figshare.25975198.v1
- Bures R. *et al.*, *Scientific Data* **12** (2025) 8. https://doi.org/10.1038/s41597-024-04286-w

## Interface languages

The entire interface is available in seven languages, with English as the default:

🇬🇧 English 🇪🇸 Espagnol 🇫🇷 Français 🇩🇪 Deutsch 🇨🇳 Chinese 🇯🇵 Japanese 🇰🇷 Korean

The language can be changed at any time from the interface.

## Roadmap

- [ ] Neural network models
- [ ] Classification tasks
- [ ] Validation on additional materials systems and target properties
- [ ] Numerical fatigue datasets (collaboration with the Department of Mechanical Engineering, Tokyo Denki University, Japan)
- [ ] Tribological test datasets
- [ ] Pedagogical assessment of installation and use by students

## Contributing

Issues and pull requests are welcome.

## Citation

If ArborML contributes to work you publish, please cite the software paper:

```bibtex
@article{Sadat_ArborML,
  author  = {Sadat, Tarik},
  title   = {ArborML: A Python-based GUI application for tree-based
             machine learning prediction of material properties},
  year    = {2026},
  note    = {Software available at https://github.com/arborml/arborml}
}
```

<!Publication submitted to SoftwareX  -->

### Related work using the ArborML analytical core

1. T. Sadat, *Prediction of Concrete Peak Load and Compressive Failure Strength Using Machine Learning*, Key Eng. Mater. **938** (2022) 235-245. https://doi.org/10.4028/p-crmx3f
2. T. Sadat, *Predicting the Average Composition of an AlFeNiTiVZr-Cr Alloy with Machine Learning and X-ray Spectroscopy*, Compounds **3** (2023) 224-232. https://doi.org/10.3390/compounds3010018
3. T. Sadat, *Machine Learning-Assisted Tensile Modulus Prediction for Flax Fiber/Shape Memory Epoxy Hygromorph Composites*, Appl. Mech. **4** (2023) 752-762. https://doi.org/10.3390/applmech4020038
4. T. Sadat, *A comparative study of machine learning approaches for predicting viscosity in Sacran/CNF solutions*, Chem. Phys. Lett. **836** (2024) 141022. https://doi.org/10.1016/j.cplett.2023.141022
5. T. Sadat, *A machine learning approach to predicting resonance frequency in soft magnetic composites*, Appl. Phys. A **132** (2026) 5. https://doi.org/10.1007/s00339-025-09175-6

## License

Distributed under the **GNU General Public License (GPL)**. See [`LICENSE`](LICENSE) for details.

ArborML is academic open-source software and is not currently used in a commercial setting.

## Contact

**Tarik Sadat**
Laboratoire d'Automatique, de Mécanique et d'Informatique Industrielles et Humaines (LAMIH), UMR CNRS 8201
Université Polytechnique Hauts-de-France, Valenciennes 59313, France

[tarik.sadat@uphf.fr](mailto:tarik.sadat@uphf.fr)

---

<div align="center">
<sub>Built for experimentalists who log their results in a spreadsheet.</sub>
</div>

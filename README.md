
# Toronto Urban Traffic Anomaly Detection

A machine-learning project for identifying unusual traffic patterns in Toronto using historical urban traffic data and unsupervised anomaly-detection techniques.

## Overview

Urban traffic conditions can change because of collisions, construction, weather, special events, road closures, and other unexpected factors. This project explores how data-mining and machine-learning methods can be used to detect traffic observations that differ significantly from normal patterns.

The repository contains:

- A Jupyter Notebook with the complete analysis workflow.
- A project report describing the problem, methodology, experiments, and findings.
- Exploratory analysis and visualizations of Toronto traffic data.
- Data preparation, feature analysis, model development, and anomaly interpretation.

## Project objectives

- Understand traffic patterns in Toronto.
- Prepare and analyze traffic-related data.
- Identify observations that may represent unusual or abnormal traffic conditions.
- Demonstrate an end-to-end machine-learning workflow in a reproducible notebook.
- Provide a foundation for future real-time traffic monitoring and intelligent transportation applications.

## Repository structure

```text
.
├── PMML_Project_Codefile.ipynb
├── Omkumar M. Patel Project Report PMML.pdf
└── README.md
```

- `PMML_Project_Codefile.ipynb` — Main notebook containing data analysis, preprocessing, modelling, visualizations, and results.
- `Omkumar M. Patel Project Report PMML.pdf` — Detailed project report and discussion of the approach and outcomes.

## Getting started

### Prerequisites

- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
- A Python environment such as `venv`, Conda, or Google Colab

### Installation

Clone the repository:

```bash
git clone [https://github.com/PATELOM925/Toronto-Urban-Traffic-Anomaly-Detection.git](https://github.com/PATELOM925/Toronto-Urban-Traffic-Anomaly-Detection.git)
cd Toronto-Urban-Traffic-Anomaly-Detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the notebook dependencies:

```bash
pip install --upgrade pip
pip install jupyter pandas numpy scipy scikit-learn matplotlib seaborn
```

If the notebook displays missing-package errors, install the package named in the error and restart the kernel.

## Running the project

Start Jupyter:

```bash
jupyter notebook
```

Open `PMML_Project_Codefile.ipynb` and run the cells from top to bottom. The notebook performs the project workflow and generates the relevant analysis outputs and visualizations.

You can also open the notebook in Google Colab by uploading the `.ipynb` file and updating any local file paths required by the data-loading cells.

## Methodology

The analysis follows a typical anomaly-detection workflow:

1. Load and inspect the traffic data.
2. Clean the data and handle missing or unsuitable values.
3. Explore distributions, relationships, and traffic patterns.
4. Prepare features for machine-learning models.
5. Train an anomaly-detection approach.
6. Examine detected anomalous observations.
7. Interpret the results in the context of urban traffic management.

The notebook is the authoritative source for the exact algorithms, parameters, feature names, and preprocessing steps used in the experiments.

## Results and interpretation

An anomaly detected by the model is an observation that differs from the learned traffic patterns. It should be treated as a candidate event for further investigation rather than automatic proof of an incident. Practical deployment would require validation against external information such as collision records, road-work data, weather, events, and traffic-camera or sensor data.

For the detailed results, assumptions, limitations, and discussion, see the project report: [`Omkumar M. Patel Project Report PMML.pdf`](./Omkumar%20M.%20Patel%20Project%20Report%20PMML.pdf).

## Limitations

- The quality of the results depends on the completeness and accuracy of the underlying traffic data.
- Anomalies may reflect data-quality issues rather than real-world traffic events.
- Historical patterns may not represent current conditions.
- Unsupervised detection does not automatically explain the cause of an anomaly.
- Reproducibility may require access to the original dataset and the same package versions used during development.

## Future improvements

- Add a versioned public dataset or a documented data-download process.
- Pin dependencies in a `requirements.txt` or `environment.yml` file.
- Compare multiple anomaly-detection methods and tune their parameters.
- Add quantitative evaluation using labelled incidents or expert validation.
- Build interactive maps and dashboards for detected anomalies.
- Add automated tests and a reproducible pipeline.
- Extend the workflow toward near-real-time traffic monitoring.

## Academic context

This project was developed as a practical machine-learning and data-mining study of urban traffic anomaly detection. Please consult the included report for the complete academic context and references.

## Author

**Omkumar M. Patel**

GitHub: [@PATELOM925](https://github.com/PATELOM925)

## License

No license has currently been specified for this repository. Unless a license is added, the code and report should not be assumed to be freely reusable or redistributable.

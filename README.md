# Gene Expression Analysis Project

This project is focused on the analysis of gene expression data, utilizing a combination of data preprocessing, dimensionality reduction, classification, and feature selection techniques. The project leverages Python and several key libraries such as `pandas`, `scikit-learn`, `torch`, and `xgboost`.

## Project Structure

```
Code/
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── autoencoder.py
│   ├── xgboost_classifier.py
│   ├── fast_recursive_feature_elimination.py
│   └── main.py
│
└── README.md
```
## Data Acquisition

Due to licensing issues, users must acquire the required datasets themselves:

1. **ADNI Data**: Apply for access on the [ADNI website](https://adni.loni.usc.edu).
2. **AddNeuroMed1 (ANM1) Data**: Download from [GEO: GSE63060](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63060).
3. **AddNeuroMed2 (ANM2) Data**: Download from [GEO: GSE63061](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63061).

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages: `pandas`, `numpy`, `scikit-learn`, `torch`, `xgboost`

You can install the necessary packages using pip:

```sh
pip install pandas numpy scikit-learn torch xgboost
```

### Running the Project

1. Ensure that your gene expression data files (e.g., `adni_gene_expression.csv`, `anm1_gene_expression.csv`, `anm2_gene_expression.csv`) and labels (e.g., `labels.csv`) are located in the same directory as the scripts.

2. Run the `main.py` script to execute the full workflow:

```sh
python main.py
```

### Output

- Preprocessed data will be saved as `preprocessed_gene_expression.csv`.
- The console will display training progress, classification accuracy, and feature importance results.

## Contribution

Feel free to fork this repository and submit pull requests if you wish to contribute or improve this project.

## License

This project is licensed under the MIT License.

---

This README provides a comprehensive overview of the gene expression analysis project, covering the purpose, structure, scripts, and usage instructions.

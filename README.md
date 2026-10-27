# Aegis-Net: final-model inference

This package provides the final model underlying Aegis-Net and its inference code. Within the integrated Aegis workflow, Aegis-Net interprets fluorescence kinetics from the first 6 min of the reaction. This is research software, not a validated clinical decision-making product.

## Contents

```text
Aegis_GitHub/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- models/
|   `-- aegis-net.pkl
|-- src/
|   |-- predict.py
|   `-- aegis_net.py
|-- examples/
|   |-- synthetic_n6.csv
|   `-- synthetic_expected.csv
`-- tests/
    `-- test_release.py
```

The single model file contains a fitted random forest, preprocessing parameters, three neural-network state dictionaries and inference metadata. Its version is `aegis-net`, and its positive-call probability cutoff is **0.593284010887146**. No retraining is required to run predictions.

`src/aegis_net.py` provides the model architecture and feature extraction; `src/predict.py` loads the model and generates predictions. This package supports final-model inference and does not include model training, baseline comparisons or original study data.

## Environment

The original analysis environment used Python 3.10.19. Create a separate Python 3.10 environment and install the inference dependencies from this directory:

```text
python -m pip install -r requirements.txt
```

To use a PyTorch build with CUDA 13.0 support, run:

```text
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130
```

## Input and prediction

Run commands from this directory. CSV and XLSX input must contain exactly the following columns, optionally preceded by `sample_id`:

```text
RFU_1,RFU_2,RFU_3,RFU_4,RFU_5,RFU_6
```

Each row represents one reaction, with readings at minutes 1 through 6 and no time-zero value. Use the original relative fluorescence unit (RFU) values; do not normalize externally. Exclude reference labels from prediction input. Empty inputs, missing or repeated sample IDs, non-numeric or non-finite RFU values and incorrect columns are rejected. If sample IDs are omitted, sequential IDs starting at 1 are generated.

```text
python src/predict.py --input examples/synthetic_n6.csv --output results/example_predictions.csv
python src/predict.py --input your_input.xlsx --output results/your_predictions.csv
```

The output CSV contains the following columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | Input ID, or generated sequential ID |
| `probability_positive` | Ensemble positive-class probability |
| `prediction` | 1 if probability is at least the stored cutoff; otherwise 0 |
| `prediction_label` | Positive or Negative |
| `probability_cutoff` | Full-precision cutoff stored in the model |
| `model_version` | Version identifier stored in the model |

Existing output files are not overwritten; use a new output path for each run. Pickle files can execute code when loaded, so use only a model from a trusted source.

## Check that the software runs correctly

```text
python -m unittest discover -s tests -v
python src/predict.py --help
```

The tests check example predictions, input validation, feature calculations and CLI functionality. An `OK` result means all tests passed. The tests require no actual study data or network access. They check software functionality, not detection performance on real samples.

`examples/synthetic_n6.csv` contains four synthetic fluorescence curves with six time points each; `examples/synthetic_expected.csv` contains the corresponding reference CPU predictions. These examples are for demonstration and testing only and do not represent actual study data or biological ground truth.

## Licence

No licence is currently included in this package. Contact the authors to confirm permission before using or redistributing the code and model.

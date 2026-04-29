# Data Privacy and Availability

This public code package does not include patient-level clinical data, real train/validation splits, raw MRI images, whole-slide images, WSI feature bags, model checkpoints, or private MRI encoder pretrained weights.

Users should prepare their own de-identified data with the schemas shown in:

- `RPC_CLI/datasets_csv/metadata/example_labels.csv`
- `RPC_CLI/datasets_csv/metadata/example_clinical.csv`
- `RPC_CLI/splits/example_5fold/rpc_example/splits_0.csv`

The example files are synthetic and are provided only to document the expected column names and folder layout.

Private or controlled-access data should remain outside the Git repository and should be supplied at runtime through command-line arguments or environment variables. The `.gitignore` file is configured to exclude common raw data, feature, checkpoint, and result file patterns.

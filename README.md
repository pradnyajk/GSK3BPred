# GSK3BPred

- **GSK3BPred** is a Deep Neural network-based model for prediction of Glycogen synthase kinase-3 beta (GSK3β) inhibitors using Simplified Molecular Input Line Entry System (SMILES) notation of molecules.
- This repository provides a Docker-based deep learning pipeline to classify whether a given molecule is inhibitor or non-inhibitor for GSK3β. It includes trained models and an Python script to make reproducible predictions in any environment using Docker.

## Repository Contents

The files contained in this repository are as follows:
- `training_scripts` : Folder contains ML and DL models training scripts 
- `Dockerfile` : Docker environment specification 
- `prediction_script_gsk3bpred.py` : Python script that loads pre-trained models and performs predictions
- `mordred_scaler.pkl`: Normalization model
- `mordred_dnn_model.h5` : DNN prediction model
- `dependencies.txt` : Dependencies to run prediction script   
- `sample.csv`: Sample of user input structures (multiple)
- `X_train.csv`: Training dataset

## Prerequisites

- Install [Docker](https://www.docker.com/) on your system.
- Input Data
	- Input data must follow the specified format  
    - A reference input file (`sample.csv`) is provided

## Input Data Format
In order to run GSK3β inhibitor predictions, save input structures as SMILES in a single file with file name as ``sample.csv``. There should be only one column in the csv file with the heading 'SMILES'. 
> **_NOTE:_** Refer to `sample.csv`

## Usage

Users have two options to use this tool:

- **Use Prebuilt Docker Image** (via GitHub Container Registry)  
- **Build Docker Image Locally** (from this repository)

### Option 1: Use Prebuilt Image
```bash
docker run --rm -v "${PWD}:/WorkPlace" ghcr.io/tanmaykumarvarma/gsk3bpred:latest sample.csv
```
Replace sample_data.csv with your input file

### ***OR***

### Option 2: Build Image Locally
### 1. Clone the Repository
```bash
git clone https://github.com/PGlab-NIPER/GSK3BPred.git
cd GSK3BPred && git lfs pull
```
> **_NOTE:_**  If git-lfs is not installed on your system, install it first and initialize it with: `git lfs install`

### 2. Build the Docker Image
```bash
docker build -t gsk3bpred .
```
### 3. Run the Prediction
#### On Linux/Windows/macOS terminal
```bash
docker run --rm -v "${PWD}:/WorkPlace" gsk3bpred sample.csv

```
> **_NOTE:_**  Replace sample.csv with your actual .csv file.

## Prediction Results
After execution, a file named ``Prediction_Results.csv`` will be saved in the same directory as your input file. It includes:
* Predicted class 
* Probability of Inhibitor
* Confidance of Predicted Class

## Python Environment Inside Docker
The Docker container uses the following Python configuration:
- python (3.12.9)
- pandas (2.2.3)
- numpy (1.26.4)
- joblib (1.4.2)
- scikit-learn (1.6.1)
- mordred (1.2.0)
- rdkit (2024.9.5)
- tensorflow (2.18.0)
- keras (3.8.0)

## Citation
If you use  **GSK3BPred** in your publication, consider citing the [paper](https://link.springer.com/article/10.1007/s11030-025-11320-5):
```
Varma, T., Kamble, P., Rajkumar, R. et al. Computational discovery of ATP-competitive GSK3β inhibitors using database-driven virtual screening and deep learning. Mol Divers (2025). https://doi.org/10.1007/s11030-025-11320-5
```

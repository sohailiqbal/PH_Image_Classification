
This project leverages topological data analysis (TDA) using persistent homology to extract features from data, aiming to classify images effectively. The package uses Ripser to compute persistent homology but can achieve faster computation times with Ripser++ if large datasets are involved.

Installation
To use this code, ensure you have MATLAB installed with Python support, for details see requirements.txt.
Usage for Covid-19 classification: Setting Up Data: Place COVID and Non-COVID images in the Data/Covid and Data/NonCovid folders, respectively.
Running the Code: Run the main MATLAB script to compute persistent homology features and classify images.

Cross-Validation: The code uses 5-fold cross-validation, with each fold dynamically sized to 20% of the dataset, making the code adaptable to any dataset size. 

Note: Since transductive approaches (like this one) require more RAM compared to inductive methods since they process both training and test data so for quick results we recommend high RAM.

Citation Requirement: If you use this code in your research, please cite the following publication:
Iqbal S, Ahmed HF, Qaiser T, Qureshi MI, Rajpoot N. Classification of COVID-19 via Homology of CT-SCAN. arXiv preprint arXiv:2102.10593. 2021 Feb 21.

License: This project is licensed under the MIT License. Please refer to the LICENSE file for details.

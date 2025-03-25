#Persistent Homology Based Image Classificationhttps://github.com/sohailiqbal/PH_Image_Classification/blob/main/README.md

This project utilizes Topological Data Analysis (TDA) through Persistent Homology to extract robust, shape-based features from images for effective classification. Persistent Homology is computed using the Ripser library, with the option to switch to Ripser++ for significantly faster performance on large datasets. The framework also supports direct comparisons with traditional computer vision techniques such as Histogram of Oriented Gradients (HOG) and Gray-Level Co-Occurrence Matrix (GLCM), providing a comprehensive evaluation of TDA against standard approaches.


Installation
### MATLAB
- MATLAB R2019b or later
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
See `requirements_matlab.txt` for MATLAB requirements.
### Python (for Ripser)
To compute persistent homology via Ripser (Python), install the following:
pip install ripser numpy scipy
To use this code, ensure you have MATLAB installed with Python support, for details see requirements.txt.


This repository provides MATLAB code for classifying COVID-19 from CT-scan images using Persistent Homology and SVM.
## 📁 Folder Structure
- `Data/Covid/` - Folder containing COVID CT images (e.g., `.png` format)
- `Data/NonCovid/` - Folder containing Non-COVID CT images
- `main.m` - Main pipeline for persistent homology + SVM classification
- `hog_classification.m` - Baseline comparison using HoG features
- `glcm_classification.m` - Baseline comparison using GLCM features
- `requirements.txt` - Python dependencies for using Ripser
- `requirements_matlab.txt` - MATLAB toolbox dependencies

.
├── main.m                          # Main pipeline (TDA + SVM)
├── hog_classification.m           # HOG feature extraction + SVM
├── glcm_classification.m          # GLCM feature extraction + SVM
├── compute_phi_images.m           # TDA helper for PH feature extraction
├── compute_lower_star_filtration.m# TDA helper for lower-star filtration
├── Data/
│   ├── Covid/                     # COVID CT images (.png)
│   └── NonCovid/                 # Non-COVID CT images
├── requirements.txt              # Python dependencies
├── requirements_matlab.txt       # MATLAB toolbox dependencies
├── LICENSE
└── README.md


##How to Run
1. Clone the repository.
2. Place images from one class in `Covid` and images from the second class in `NonCovid`.
3. In MATLAB, run: main

Cross-Validation: The code uses 5-fold cross-validation, with each fold dynamically sized to 20% of the dataset, making the code adaptable to any dataset size. 
Note: Since transductive approaches (like this one) require more RAM compared to inductive methods since they process both training and test data so for quick results we recommend high RAM.


Citation Requirement: If you use this code in your research, please cite the following publication:
Iqbal S, Ahmed HF, Qaiser T, Qureshi MI, Rajpoot N. Classification of COVID-19 via Homology of CT-SCAN. arXiv preprint arXiv:2102.10593. 2021 Feb 21.

License: This project is licensed under the MIT License. Please refer to the LICENSE file for details.




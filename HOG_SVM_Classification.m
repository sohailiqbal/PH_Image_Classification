clc;
clear;
close all;

% --------------------------------------------
% Set up parameters for HOG feature extraction
% --------------------------------------------

% Define a consistent image size to resize all images to 250x250 pixels
imageSize = [250 250];    % Resize all images to 250x250
pcaComponents = 4800;     % Number of principal components to retain after PCA for dimensionality reduction

% --------------------------------------------
% Directory Setup for COVID and Non-COVID Images
% --------------------------------------------

% Define the directories for COVID and Non-COVID images
dataDir_Covid = fullfile(pwd, 'Covid');
dataDir_NonCovid = fullfile(pwd, 'NonCovid');

% Get a list of all .png image files in each directory
covidFiles = dir(fullfile(dataDir_Covid, '*.png'));
nonCovidFiles = dir(fullfile(dataDir_NonCovid, '*.png'));

% Initialize variables to store HOG features and labels
hogFeaturesAll = [];  % Array to store HOG features for all images
labels = [];          % Array to store labels (1 for Non-COVID, 2 for COVID)

% --------------------------------------------
% Extract HOG Features for COVID Images
% --------------------------------------------

% Loop through each COVID image to extract HOG features
for i = 1:length(covidFiles)
    % Read and resize each image
    imagePath = fullfile(dataDir_Covid, covidFiles(i).name);
    image = imread(imagePath);
    image = imresize(image, imageSize);  % Resize image to 250x250 pixels
    
    % Convert to grayscale if the image is in RGB format
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    
    % Extract HOG features with a cell size of 8x8
    hogFeatures = extractHOGFeatures(image, 'CellSize', [8 8]);
    
    % Store the extracted features and assign label 2 for COVID
    hogFeaturesAll = [hogFeaturesAll; hogFeatures];
    labels = [labels; 2];  % Label 2 represents COVID
end

% --------------------------------------------
% Extract HOG Features for Non-COVID Images
% --------------------------------------------

% Loop through each Non-COVID image to extract HOG features
for i = 1:length(nonCovidFiles)
    % Read and resize each image
    imagePath = fullfile(dataDir_NonCovid, nonCovidFiles(i).name);
    image = imread(imagePath);
    image = imresize(image, imageSize);  % Resize image to 250x250 pixels
    
    % Convert to grayscale if the image is in RGB format
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    
    % Extract HOG features with a cell size of 8x8
    hogFeatures = extractHOGFeatures(image, 'CellSize', [8 8]);
    
    % Store the extracted features and assign label 1 for Non-COVID
    hogFeaturesAll = [hogFeaturesAll; hogFeatures];
    labels = [labels; 1];  % Label 1 represents Non-COVID
end

% --------------------------------------------
% Convert Labels to Categorical for SVM
% --------------------------------------------

labels = categorical(labels);  % Convert numerical labels to categorical for compatibility with SVM

% --------------------------------------------
% Apply PCA for Dimensionality Reduction
% --------------------------------------------

% Perform PCA on the HOG features, reducing dimensions to pcaComponents
[coeff, hogFeaturesAllPCA] = pca(hogFeaturesAll, 'NumComponents', pcaComponents);

% --------------------------------------------
% Split Data into Training and Test Sets (80/20 Split)
% --------------------------------------------

% Create an 80/20 split for training and testing
cv = cvpartition(labels, 'HoldOut', 0.2);
X_trainT_P = hogFeaturesAllPCA(training(cv), :);  % Training data
y_trainT_P = labels(training(cv));                % Training labels
X_testT_P = hogFeaturesAllPCA(test(cv), :);       % Test data
y_testT_P = labels(test(cv));                     % Test labels

% --------------------------------------------
% Train SVM Model with Optimized Hyperparameters
% --------------------------------------------

% Set random seed for reproducibility
rng('default');

% Train the SVM model with RBF kernel and optimize hyperparameters
mdl = fitcsvm(X_trainT_P, y_trainT_P, 'KernelFunction', 'rbf', ...
              'OptimizeHyperparameters', 'auto', ...
              'HyperparameterOptimizationOptions', ...
              struct('AcquisitionFunctionName', 'expected-improvement-plus'));

% --------------------------------------------
% Evaluate Model on Test Set
% --------------------------------------------

% Predict labels for test data
y_pred = predict(mdl, X_testT_P);

% --------------------------------------------
% Calculate Performance Metrics
% --------------------------------------------

% Calculate accuracy
accuracy = sum(y_pred == y_testT_P) / numel(y_testT_P);

% Generate confusion matrix
confusionMat = confusionmat(y_testT_P, y_pred);

% Calculate precision, recall, and F1 score
precision = confusionMat(2,2) / sum(confusionMat(:,2));
recall = confusionMat(2,2) / sum(confusionMat(2,:));
f1Score = 2 * (precision * recall) / (precision + recall);

% --------------------------------------------
% Display Results
% --------------------------------------------

% Print performance metrics
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1Score);

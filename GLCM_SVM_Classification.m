clc;
clear;
close all;

% --------------------------------------------
% Define Parameters for Image Preprocessing
% --------------------------------------------

% Define a consistent image size to resize all images to 250x250 pixels
imageSize = [250 250];   % Target image size
glcmFeaturesAll = [];    % Array to store extracted GLCM features
labels = [];             % Array to store image labels (1 for NonCovid, 2 for Covid)

% --------------------------------------------
% Directory Setup for COVID and Non-COVID Images
% --------------------------------------------

% Set up directories for COVID and Non-COVID images
dataDir_Covid = fullfile(pwd, 'Covid');
dataDir_NonCovid = fullfile(pwd, 'NonCovid');

% List all .png image files in each directory
covidFiles = dir(fullfile(dataDir_Covid, '*.png'));
nonCovidFiles = dir(fullfile(dataDir_NonCovid, '*.png'));

% --------------------------------------------
% Extract GLCM Features for COVID Images
% --------------------------------------------

% Loop through each COVID image to extract GLCM features
for i = 1:length(covidFiles)
    % Read and resize the image
    imagePath = fullfile(dataDir_Covid, covidFiles(i).name);
    image = imread(imagePath);
    image = imresize(image, imageSize);  % Resize image to 250x250 pixels
    
    % Convert to grayscale if the image is RGB
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    
    % Compute GLCM for 1-pixel distance in four directions (0°, 45°, 90°, and 135°)
    glcm = graycomatrix(image, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
    
    % Extract texture properties from GLCM
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    % Calculate mean values across directions for each property and store
    glcmFeatures = [mean(stats.Contrast), mean(stats.Correlation), ...
                    mean(stats.Energy), mean(stats.Homogeneity)];
                
    glcmFeaturesAll = [glcmFeaturesAll; glcmFeatures];
    labels = [labels; 2];  % Label 2 represents COVID
end

% --------------------------------------------
% Extract GLCM Features for Non-COVID Images
% --------------------------------------------

% Loop through each Non-COVID image to extract GLCM features
for i = 1:length(nonCovidFiles)
    % Read and resize the image
    imagePath = fullfile(dataDir_NonCovid, nonCovidFiles(i).name);
    image = imread(imagePath);
    image = imresize(image, imageSize);  % Resize image to 250x250 pixels
    
    % Convert to grayscale if the image is RGB
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    
    % Compute GLCM for 1-pixel distance in four directions (0°, 45°, 90°, and 135°)
    glcm = graycomatrix(image, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
    
    % Extract texture properties from GLCM
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    % Calculate mean values across directions for each property and store
    glcmFeatures = [mean(stats.Contrast), mean(stats.Correlation), ...
                    mean(stats.Energy), mean(stats.Homogeneity)];
                
    glcmFeaturesAll = [glcmFeaturesAll; glcmFeatures];
    labels = [labels; 1];  % Label 1 represents NonCovid
end

% --------------------------------------------
% Prepare Data for SVM Classification
% --------------------------------------------

% Convert labels to categorical format for SVM compatibility
labels = categorical(labels);

% Split data into training and test sets with an 80/20 split
cv = cvpartition(labels, 'HoldOut', 0.2);
X_trainT_P = glcmFeaturesAll(training(cv), :);  % Training data
y_trainT_P = labels(training(cv));               % Training labels
X_testT_P = glcmFeaturesAll(test(cv), :);        % Test data
y_testT_P = labels(test(cv));                    % Test labels

% --------------------------------------------
% Train the SVM Model with Hyperparameter Optimization
% --------------------------------------------

% Set random seed for reproducibility
rng('default');  % Ensures reproducible results

% Train the SVM model with RBF kernel and optimize hyperparameters
mdl = fitcsvm(X_trainT_P, y_trainT_P, 'KernelFunction', 'rbf', ...
              'OptimizeHyperparameters', 'auto', ...
              'HyperparameterOptimizationOptions', ...
              struct('AcquisitionFunctionName', 'expected-improvement-plus'));

% --------------------------------------------
% Evaluate Model Performance on Test Set
% --------------------------------------------

% Predict labels for the test set
y_pred = predict(mdl, X_testT_P);

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

% Print accuracy, precision, recall, and F1 score
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1Score);

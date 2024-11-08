clc;
clear;
close all;

% --------------------------------------------
%      Persistent Homology Libraries Setup
% --------------------------------------------

% You have two options for calculating persistent homology:
% 1. JavaPlex (a MATLAB library)
% 2. Ripser (a fast C++ library accessed via Python)
%   (see this for details: https://www.mathworks.com/help/matlab/matlab_external/create-object-from-python-class.html)

% Option 1: Using JavaPlex
% -------------------------
% Uncomment the following lines to use JavaPlex for persistent homology.
% Ensure that JavaPlex is installed and added to your MATLAB path.

% load_javaplex;
% import edu.stanford.math.plex4.*;

% Option 2: Using Ripser via Python
% ---------------------------------
% To use Ripser, ensure that Python and the Ripser library are installed.
% MATLAB must be configured to use the correct Python version.
% For instructions on setting up Ripser with MATLAB, see the README section below.


% --------------------------------------------
%      Add External Libraries and Toolboxes
% --------------------------------------------
%download from the link :
%https://github.com/rushilanirudh/pdsphere/tree/master/matlab
addpath(genpath('libsvm-3.21'));     
addpath(genpath('Sphere tools'));    


% --------------------------------------------
%   Number of Initial Dimensions for PGA
% --------------------------------------------

k = 2400;                            % Number of dimensions for PGA (Principal Geodesic Analysis)


% --------------------------------------------
%              Ripser Parameter Values
% --------------------------------------------
%We use ripser library for our computations
max_dimension = 2;                   % Max homology dimension to compute
max_filtration_value = 500;          % Max filtration value for persistence
Finite_Field = 7;                    % Finite field for coefficient calculations
% Create a Ripser object with specified parameters
rips = py.ripser.Rips('maxdim', max_dimension, 'thresh', max_filtration_value, 'coeff', Finite_Field);

% Note: To switch between JavaPlex and Ripser, comment or uncomment the relevant sections.
% Adjust subsequent code to use the functions and data formats of the chosen library.

% --------------------------------------------
%        Clustering and K-means Parameters
% --------------------------------------------
% the following parameters controll the number of points in teh point
% cloud
NkClusters = 10;                     % Number of clusters for k-means clustering
d=0.9                                % shape parameter

% --------------------------------------------
%         Parameters for Gaussian Filter
% --------------------------------------------

sig = 0.2;                           % Standard deviation for Gaussian
x1 = 0:0.5:max_filtration_value+30;  % Grid range for x-axis
x2 = 0:0.5:max_filtration_value+30;  % Grid range for y-axis
Sigma = [sig 0; 0 sig];              % Covariance matrix for Gaussian filter
% =======================================================
% MAIN PROCESSING: Load Images and Compute Persistent Homology
% =======================================================
% Define paths to COVID and Non-COVID image directories
% Define paths to COVID and Non-COVID image directories
dataDir_NonCovid = fullfile('Data/NonCovid');   % Path to Non-COVID images
dataDir_Covid = fullfile('Data/Covid');         % Path to COVID images

% Get directory listing for both Non-COVID and COVID images
nonCovidFiles = dir(fullfile(dataDir_NonCovid, '*.png'));  % List all PNG images (modify extension if needed)
covidFiles = dir(fullfile(dataDir_Covid, '*.png'));        % List all PNG images (modify extension if needed)

% Filter out hidden files or irrelevant directories (if any)
nonCovidFiles = nonCovidFiles(~[nonCovidFiles.isdir]);  % Exclude directories
covidFiles = covidFiles(~[covidFiles.isdir]);

% Display the number of images (optional, for debugging purposes)
fprintf('Number of Non-COVID images: %d\n', length(nonCovidFiles));
fprintf('Number of COVID images: %d\n', length(covidFiles));

% Compute phi for Non-COVID images
phi_N = compute_phi_images(dataDir_NonCovid, NkClusters,d, rips, max_dimension, max_filtration_value, x1, x2, Sigma);

% Compute phi for COVID images
phi_C = compute_phi_images(dataDir_Covid, NkClusters,d, rips, max_dimension, max_filtration_value, x1, x2, Sigma);

%----------------Lower Star Filtration -----------------------------
% Warning: Lower star filtration can not be computed using JavaPlex. 
% Define parameters
a = 0:0.3:256 + 20;
b = 0:0.3:256 + 20;
% Compute lower star filtration for COVID images
phi_LSF_C = compute_lower_star_filtration(dataDir_Covid, a, b, Sigma);
% Compute lower star filtration for Non-COVID images
phi_LSF_N= compute_lower_star_filtration(dataDir_NonCovid, a, b, Sigma);
% -----------------End of lower Star filtration-----------------------
phi_C(:,4)=phi_LSF_C
phi_N(:,4)=phi_LSF_N
%-----------------------LoadMatricesHere------------------------

% Combine phi_N and phi_C into phi
phi = [phi_N; phi_C];

% Generate labels for Non-COVID and COVID images
label = [ones(size(phi_N, 1), 1); 2 * ones(size(phi_C, 1), 1)];

% Preallocate PGAPatches based on total number of images and dimensions
[NofImage, Hj] = size(phi);
PGAPatches = zeros(NofImage, Hj * k);

% Compute PGAPatches for each dimension
for n = 1:Hj
    % Compute extrinsic mean for dimension n
    mu = sphere_extrinsic_mean(phi(:, n));
    
    % Precompute log_map for all images at once
    V = zeros(size(phi{1, n}, 1), NofImage); % Assuming each phi{m, n} has the same size
    for m = 1:NofImage
        V(:, m) = log_map(mu, phi{m, n});
    end
    
    % Perform PCA and store results in PGAPatches
    [~, w, ~] = pca(V');
    PGAPatches(:, (n - 1) * k + 1 : n * k) = w(:, 1:k);
end

% Perform PCA on PGAPatches and reduce dimensions
[~, w, ~] = pca(PGAPatches');
PGAPatchesT = w(:, 1:4800)';  % Transpose after slicing to get reduced dimensions

% Parameters for dataset and model
[NoP, par] = size(PGAPatchesT);
mNoP = min(sum(label == 1), sum(label == 2));
NoPtest = 2 * ceil(0.2 * mNoP);  % Calculate test size dynamically based on data size
NoPtrain = NoP - NoPtest;
d = 4800;

% Randomly permute the data indices for cross-validation
rand_perm_T = randperm(NoP);
num_folds = 5;
fold_size = NoPtest;

% Initialize arrays to store metrics for each fold
Precision = zeros(num_folds, 1);
Recall = zeros(num_folds, 1);
Specificity = zeros(num_folds, 1);
F1 = zeros(num_folds, 1);
Accuracy = zeros(num_folds, 1);

% Perform 5-fold cross-validation
for fold = 1:num_folds
    % Define test and train indices for this fold
    test_start = (fold - 1) * fold_size + 1;
    test_end = fold * fold_size;
    
    % Select test indices for this fold
    test_idx = rand_perm_T(test_start:test_end);
    
    % Train indices are all other indices not in the test fold
    train_idx = setdiff(rand_perm_T, test_idx);
    
    % Split into train and test sets
    X_trainT_P = PGAPatchesT(train_idx, :);
    y_trainT_P = label(train_idx);
    X_testT_P = PGAPatchesT(test_idx, :);
    y_testT_P = label(test_idx);
    
    % Train SVM model with RBF kernel and optimized hyperparameters
    rng('default');
    mdl = fitcsvm(X_trainT_P, y_trainT_P, 'KernelFunction', 'rbf', ...
                  'OptimizeHyperparameters', 'auto', ...
                  'HyperparameterOptimizationOptions', ...
                  struct('AcquisitionFunctionName', 'expected-improvement-plus'));
    
    % Initialize counters for TP, TN, FP, FN
    TP = 0; TN = 0; FP = 0; FN = 0;
    
    % Evaluate on test set
    for i = 1:fold_size
        actual_label = y_testT_P(i);
        predicted_label = predict(mdl, X_testT_P(i, 1:d));
        
        if actual_label == 2
            if predicted_label == 2
                TP = TP + 1;
            else
                FN = FN + 1;
            end
        elseif actual_label == 1
            if predicted_label == 1
                TN = TN + 1;
            else
                FP = FP + 1;
            end
        end
    end
    
    % Calculate metrics for this fold
    Precision(fold) = TP / (TP + FP);
    Recall(fold) = TP / (TP + FN);
    Specificity(fold) = TN / (TN + FP);
    F1(fold) = 2 * (Precision(fold) * Recall(fold)) / (Precision(fold) + Recall(fold));
    Accuracy(fold) = (TP + TN) / fold_size;
end

% Display averaged results across all folds
fprintf('Average Precision: %.4f\n', mean(Precision));
fprintf('Average Recall: %.4f\n', mean(Recall));
fprintf('Average Specificity: %.4f\n', mean(Specificity));
fprintf('Average F1 Score: %.4f\n', mean(F1));
fprintf('Average Accuracy: %.4f\n', mean(Accuracy));
 

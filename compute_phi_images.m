function phi = compute_phi_images(imagePath, NkClusters,d, rips, max_dimension, max_filtration_value, x1, x2, Sigma)
    % compute_phi_images: Computes the phi (cell array) for all images in the folder.
    %
    % Inputs:
    %   - imagePath: Path to the folder containing images (string)
    %   - NkClusters: Number of clusters for k-means clustering (integer)
    %   - rips: Ripser object for persistent homology computation
    %   - max_dimension: Maximum dimension for persistent homology (integer)
    %   - max_filtration_value: Maximum filtration value (integer)
    %   - x1, x2: Grid ranges for heat map (vectors)
    %   - Sigma: Covariance matrix for Gaussian filter (matrix)
    %
    % Outputs:
    %   - phi: Cell array containing the resulting phi for each image and dimension
    
    % Get list of all image files in the specified directory
    imageFiles = dir(fullfile(imagePath, '*.png'));  % Modify this if images are not PNG
    numImages = length(imageFiles);                 % Number of images found
    
    % Preallocate the output cell array
    phi = cell(numImages, max_dimension + 1);
    
    % Loop through each image in the directory
    for NoP = 1:numImages
        fprintf('Processing image %d of %d...\n', NoP, numImages);  % Display progress

        % Load image using the file name from the directory listing
        inputImg = imread(fullfile(imagePath, imageFiles(NoP).name));
        
        % Perform k-means clustering to get the point cloud
        CCmatrix = solidities_kmeans(inputImg, NkClusters,d); 
        point_cloud = py.numpy.array(CCmatrix);
        
        % Compute persistent diagrams using Ripser
        diagrams = rips.fit_transform(point_cloud);
        PDs = cell(diagrams);
        
        % Loop through each dimension to compute heat maps and phi values
        for j = 1:max_dimension + 1
            HeatMap = zeros(length(x1), length(x1));
            
            % Extract persistence diagram for the current dimension
            PersDiag = double(PDs{1, j});
            
            % Loop through each persistence pair in the diagram
            for i = 1:size(PersDiag, 1)
                mu = [PersDiag(i, 1), PersDiag(i, 2)];
                
                % Replace infinity with a large finite value
                if PersDiag(i, 2) == Inf
                    mu(2) = max_filtration_value + 30;
                end
                
                % Generate Gaussian heat map for the persistence pair
                [X1, X2] = meshgrid(x1, x2);
                F = mvnpdf([X1(:), X2(:)], mu, Sigma);
                F = reshape(F, length(x2), length(x1));
                HeatMap = HeatMap + F;
            end
            
            % Normalize and compute the phi value
            ht = HeatMap / (sum(HeatMap(:)) + 1e-10);
            if ~sum(ht(:))
                ht(1) = 1;
            end
            
            % Store the result in the phi cell array
            phi{NoP, j} = sqrt(ht(:));
        end
    end
end

function phi_LSF = compute_lower_star_filtration(imagePath, a, b, Sigma)
    % compute_lower_star_filtration: Computes lower star filtration for a set of images.
    %
    % Inputs:
    %   - imagePath: Path pattern to the image files with a placeholder (e.g., 'Covid\\Covid (%d).png')
    %   - numImages: Number of images to process
    %   - a, b: Grid range vectors for Gaussian filter
    %   - Sigma: Covariance matrix for Gaussian filter
    %
    % Outputs:
    %   - phi_LSF: Cell array of computed phi values for each image
    %   - HT_map_LSF: Cell array of computed heat maps for each image
    imageFiles = dir(fullfile(imagePath, '*.png'));  % Modify this if images are not PNG
    numImages = length(imageFiles);                 % Number of images found
    
    % Preallocate output cell arrays
    phi_LSF = cell(numImages, 1);
    HT_map_LSF = cell(numImages, 1);
    
    % Loop through each image
    for NoP = 1:numImages
        fprintf('Processing image %d of %d...\n', NoP, numImages);  % Display progress
        inputImg = imread(fullfile(imagePath, imageFiles(NoP).name));
        
        % Load and process the image
        inputImgGray = rgb2gray(inputImg);
        imarray = py.numpy.array(inputImgGray);
        
        % Compute lower star filtration using Ripser
        lsi = py.ripser.lower_star_img(imarray);
        PersDiag = double(lsi);  % Convert persistent diagram to double format
        
        % Initialize the heat map
        HeatMap = zeros(length(a), length(b));
        
        % Generate Gaussian heat map for each persistence pair
        for i = 1:size(PersDiag, 1)
            mu = [PersDiag(i, 1), PersDiag(i, 2)];
            
            % Replace infinity with a large finite value
            if PersDiag(i, 2) == Inf
                mu(2) = max(a) + 20;
            end
            
            [X1, X2] = meshgrid(a, b);
            F = mvnpdf([X1(:), X2(:)], mu, Sigma);
            F = reshape(F, length(b), length(a));
            HeatMap = HeatMap + F;
        end
        
        % Normalize and compute the phi value
        ht = HeatMap / (sum(HeatMap(:)) + 1e-10);
        if ~sum(ht(:))
            ht(1) = 1;
        end
        
        % Store results in cell arrays
        phi_LSF{NoP, 1} = sqrt(ht(:));
    end
end


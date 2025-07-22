function [corners] = detector(image_path)
    % Read and prepare the image
    if isstring(image_path) || ischar(image_path)
        img = imread(image_path);
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
    else
        img = image_path;
    end
    
    % Convert to double and normalize
    img = im2double(img);
    %img = imresize(img, 0.5);
    
    % 1. Harris Corner Detection
    corners.harris = detectHarrisCorners(img);
    
    % 2. FAST Corner Detection
    corners.fast = detectFASTCorners(img);

end

function corners = detectHarrisCorners(img)
    % Parameters for Harris corner detection
    sigma = 2;
    threshold = 0.0001;
    window_size = 5;
    
    % Compute gradients
    [Ix, Iy] = gradient(img);
    
    % Compute products of gradients
    Ix2 = Ix .* Ix;
    Iy2 = Iy .* Iy;
    Ixy = Ix .* Iy;
    
    % Apply Gaussian smoothing
    h = fspecial('gaussian', [6*sigma+1, 6*sigma+1], sigma);
    Ix2 = imfilter(Ix2, h);
    Iy2 = imfilter(Iy2, h);
    Ixy = imfilter(Ixy, h);
    
    % Compute Harris response
    k = 0.04; % Harris parameter
    R = (Ix2 .* Iy2 - Ixy.^2) - k * (Ix2 + Iy2).^2;
    
    % Apply non-maximum suppression
    R = nonMaxSuppression(R, window_size);
    
    % Threshold and find corner locations
    corners_mask = R > threshold * max(R(:));
    [y, x] = find(corners_mask);
    corners = [x, y];
end

function corners = detectFASTCorners(img)
    % FAST corner detection parameters
    N = 12;  % Number of contiguous pixels
    threshold = 0.04;
    radius = 3;
    
    % Initialize corner map
    corners_map = false(size(img));
    [rows, cols] = size(img);
    
    % Generate circle points
    circle_points = generateCirclePoints(radius);
    num_points = size(circle_points, 1);
    
    % Scan through each pixel
    for i = radius+1:rows-radius
        for j = radius+1:cols-radius
            % Get intensity values on the circle
            circle_values = zeros(num_points, 1);
            for k = 1:num_points
                y = i + circle_points(k, 1);
                x = j + circle_points(k, 2);
                circle_values(k) = img(y, x);
            end
            
            % Check if point is a corner
            center_value = img(i, j);
            if isFASTCorner(circle_values, center_value, threshold, N)
                corners_map(i, j) = true;
            end
        end
    end
    
    % Get corner coordinates
    [y, x] = find(corners_map);
    corners = [x, y];
end

% Helper Functions
function points = generateCirclePoints(radius)
    % Generate approximate circle points using Bresenham's circle algorithm
    t = 0:pi/8:2*pi;
    x = round(radius * cos(t));
    y = round(radius * sin(t));
    points = unique([y', x'], 'rows');
end

function is_corner = isFASTCorner(circle_values, center_value, threshold, N)
    % Check if point is a FAST corner
    brighter = circle_values > (center_value + threshold);
    darker = circle_values < (center_value - threshold);
    
    % Look for N contiguous pixels
    is_corner = false;
    for i = 1:length(circle_values)
        if sum(brighter(i:min(i+N-1, length(circle_values)))) >= N || ...
           sum(darker(i:min(i+N-1, length(circle_values)))) >= N
            is_corner = true;
            break;
        end
    end
end

function suppressed = nonMaxSuppression(img, window_size)
    % Apply non-maximum suppression
    suppressed = zeros(size(img));
    pad = floor(window_size/2);
    
    for i = pad+1:size(img,1)-pad
        for j = pad+1:size(img,2)-pad
            window = img(i-pad:i+pad, j-pad:j+pad);
            if img(i,j) == max(window(:))
                suppressed(i,j) = img(i,j);
            end
        end
    end
end
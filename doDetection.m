close all
clear all
clc

% For an image file
[corners] = detector('homework.jpg');

img = imread('homework.jpg');
%img = imresize(img, 0.5);

disp("Corners detected!");

% Plot
close all
fig = figure;
saveas(fig, 'output/corners_detected.png')
imshow(img);
hold on;
% Plot Harris corners
plot(corners.harris(:,1), corners.harris(:,2), 'r+');
% Plot FAST corners
plot(corners.fast(:,1), corners.fast(:,2), 'g+');
title("Detected corners")
saveas(fig, 'output/corners_detected.png')

% Enable data cursor mode
datacursormode on
dcm_obj = datacursormode(fig);
% Set update function
set(dcm_obj,'UpdateFcn',@myupdatefcn)

%% Lines
point_names = ["A", "B", "C", "D", "E", "F", "G", "H", "M", "N", "O", "P", "Q", "R"];

% Initialize points structure
pts = [];
i = 1;
while i <= length(point_names)
    disp('Select point ' + point_names(i) + ', and then press Enter')
    key = input('', 's');
    
    % if strcmpi(key, 'q')
    %     break
    % end
    
    % Export cursor to workspace
    info_struct = getCursorInfo(dcm_obj);
    if isfield(info_struct, 'Position')
        pos = info_struct.Position;
        pts(i, :) = [pos(1) pos(2) 1]
        disp(['Point ' num2str(i) ': (' num2str(pos(1)) ', ' num2str(pos(2)) ')'])
        i = i + 1;
    end
end

l1 = findLine(5,6, pts);
l2 = findLine(3,4, pts);
l3 = findLine(1,2, pts);

m1 = findLine(5,7,pts);
m2 = findLine(9,11,pts);
m3 = findLine(10,12,pts);
m4 = findLine(8,6,pts);
m5 = findLine(1,4,pts);
m6 = findLine(2,3,pts);

h1 = findLine(4,5,pts);
h2 = findLine(11,13,pts);
h3 = findLine(12,14,pts);
h4 = findLine(3,6,pts);

lx = findLine(7, 8, pts);

save('elements/lines.mat', 'l1', 'l2', 'l3', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'h1', 'h2', 'h3', 'h4', 'lx');

disp("Lines extracted!");

%% C

i = 1;
xc = [];
yc = [];
while true;
    disp('Select points for C, press q to go on')
    key = input('', 's');
    
    if strcmpi(key, 'q')
        break
    end
    
    % Export cursor to workspace
    info_struct = getCursorInfo(dcm_obj);
    if isfield(info_struct, 'Position')
        pos = info_struct.Position;
        % pts(i, :) = [pos(1) pos(2) 1]
        xc(i) = pos(1);
        yc(i) = pos(2);
        disp(['Point ' num2str(i) ': (' num2str(pos(1)) ', ' num2str(pos(2)) ')'])
        i = i + 1;
    end
end
xc = xc';
yc = yc';
scatter(xc,yc,100,'filled');
hold off;
% Estimation of conic parameters
A1=[xc.^2 xc.*yc yc.^2 xc yc ones(size(xc))];
[~,~,V1] = svd(A1);
N1 = V1(:,end);
cc1 = N1(:, 1);
% changing the name of variables
[a1, b1, c1, d1, e1, f1] = deal(cc1(1),cc1(2),cc1(3),cc1(4),cc1(5),cc1(6));
% matrix of the conic: off-diagonal elements must be divided by two
cc1=[a1 b1/2 d1/2; b1/2 c1 e1/2; d1/2 e1/2 f1];

save('elements/circumference.mat', "a1", "b1", "c1", "d1", "e1", "f1");
disp("C extracted!");

%% S
i = 1;
while true;
    disp('Select points for S, press q to go on')
    key = input('', 's');
    
    if strcmpi(key, 'q')
        break
    end
    
    % Export cursor to workspace
    info_struct = getCursorInfo(dcm_obj);
    if isfield(info_struct, 'Position')
        pos = info_struct.Position;
        % pts(i, :) = [pos(1) pos(2) 1]
        x(i) = pos(1);
        y(i) = pos(2);
        disp(['Point ' num2str(i) ': (' num2str(pos(1)) ', ' num2str(pos(2)) ')'])
        i = i + 1;
    end
end

sx = x';
sy = y';

save('elements/s_points.mat', 'sx', 'sy');
disp("S ectracted!")

function output_txt = myupdatefcn(~,event_obj)
pos = get(event_obj, 'Position');
output_txt = {['x: ' num2str(pos(1))], ['y: ' num2str(pos(2))]};
end
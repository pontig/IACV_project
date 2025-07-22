close all
clear all
clc

%% Rectified spline

load('elements/s_points_rect.mat', 'sx', 'sy'); % Rectified points of S

[x_fine, y_fine] = interpolateSpline(sx, sy);

figS  = figure;
title('Rectified curve S');
plot(sx, sy, 'o', 'MarkerSize', 8, 'DisplayName', 'Original Points'); % Original points
hold on;
plot(x_fine, y_fine, 'LineWidth', 2, 'DisplayName', 'Spline Interpolation'); % Spline curve

saveas(figS, 'output/rectified_curve_S.png');


%% 3D Views
load('elements/real_distances.mat', 'm', 'h', 'EO_real', 'EP_real');

o = EO_real;
p = EP_real;
width = 1;
depth = m;
height = h;

circle_center = [o/2 m/2 h/3];
circle_radius = m/3;

segments = [ % [x1, y1, z1, x2, y2, z2]
    o 0 0 o 0 h % OQ
    o 0 h o m h % OM
    p 0 0 p 0 h % PR
    p 0 h p m h % PN
    ];

plotParallelogramWithSegments(width, depth, height, segments, circle_center, circle_radius);

function plotParallelogramWithSegments(width, depth, height, segments, circle_center, circle_radius)

% Define the vertices of the parallelogram
vertices = [
    0, 0, 0;
    width, 0, 0;
    width, depth, 0;
    0, depth, 0;
    0, 0, height;
    width, 0, height;
    width, depth, height;
    0, depth, height;
    ];

% Define the faces of the parallelogram
faces = [
    1, 2, 3, 4; % Bottom face
    5, 6, 7, 8; % Top face
    1, 2, 6, 5; % Front face
    2, 3, 7, 6; % Right face
    3, 4, 8, 7; % Back face
    4, 1, 5, 8; % Left face
    ];

% Plot the parallelogram
plts = figure('units','normalized','outerposition',[0 0 1 1]);
% Three different views
style = 'k';
facecolor = '#978e3f';

subplot(2, 2, 1);
patch('Vertices', vertices, 'Faces', faces, 'FaceColor', facecolor, 'FaceAlpha', 0.5);
hold on;
for i = 1:size(segments, 1)
    plot3([segments(i, 1), segments(i, 4)], [segments(i, 2), segments(i, 5)], [segments(i, 3), segments(i, 6)], style);
end
 % Plot the circumference
if ~isempty(circle_center) && ~isempty(circle_radius)
    theta = linspace(0, 2*pi, 100);
    x_circle = circle_center(1) + circle_radius * cos(theta);
    y_circle = circle_center(2) + circle_radius * sin(theta);
    z_circle = circle_center(3) * ones(size(theta));
    plot3(x_circle, y_circle, z_circle, 'k');
end
hold off;
view(-45, 30);
title('View 1');
axis equal;
xlabel('X(l)');
ylabel('Y(m)');
zlabel('Z(h)');


subplot(2, 2, 2);
patch('Vertices', vertices, 'Faces', faces, 'FaceColor', facecolor, 'FaceAlpha', 0.5);
hold on;
for i = 1:size(segments, 1)
    plot3([segments(i, 1), segments(i, 4)], [segments(i, 2), segments(i, 5)], [segments(i, 3), segments(i, 6)], style);
end
% Plot the circumference
if ~isempty(circle_center) && ~isempty(circle_radius)
    theta = linspace(0, 2*pi, 100);
    x_circle = circle_center(1) + circle_radius * cos(theta);
    y_circle = circle_center(2) + circle_radius * sin(theta);
    z_circle = circle_center(3) * ones(size(theta));
    plot3(x_circle, y_circle, z_circle, 'k');
end
hold off;
view(60, 10);
title('View 2');
axis equal;
xlabel('X(l)');
ylabel('Y(m)');
zlabel('Z(h)');


subplot(2, 2, 3);
patch('Vertices', vertices, 'Faces', faces, 'FaceColor', facecolor, 'FaceAlpha', 0.5);
hold on;
for i = 1:size(segments, 1)
    plot3([segments(i, 1), segments(i, 4)], [segments(i, 2), segments(i, 5)], [segments(i, 3), segments(i, 6)], style);
end
% Plot the circumference
if ~isempty(circle_center) && ~isempty(circle_radius)
    theta = linspace(0, 2*pi, 100);
    x_circle = circle_center(1) + circle_radius * cos(theta);
    y_circle = circle_center(2) + circle_radius * sin(theta);
    z_circle = circle_center(3) * ones(size(theta));
    plot3(x_circle, y_circle, z_circle, 'k');
end
hold off;
view(-75, -45);
title('View 3');
axis equal;
xlabel('X(l)');
ylabel('Y(m)');
zlabel('Z(h)');

subplot(2, 2, 4);
patch('Vertices', vertices, 'Faces', faces, 'FaceColor', facecolor, 'FaceAlpha', 0.5);
hold on;
for i = 1:size(segments, 1)
    plot3([segments(i, 1), segments(i, 4)], [segments(i, 2), segments(i, 5)], [segments(i, 3), segments(i, 6)], style);
end
% Plot the circumference
if ~isempty(circle_center) && ~isempty(circle_radius)
    theta = linspace(0, 2*pi, 100);
    x_circle = circle_center(1) + circle_radius * cos(theta);
    y_circle = circle_center(2) + circle_radius * sin(theta);
    z_circle = circle_center(3) * ones(size(theta));
    plot3(x_circle, y_circle, z_circle, 'k');
end
hold off;
view(10, 5);
title('View 4');
axis equal;
xlabel('X(l)');
ylabel('Y(m)');
zlabel('Z(h)');

saveas(plts, 'output/final_problem.png');
end
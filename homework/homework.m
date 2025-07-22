close all
clear all
clc
warning('off','all')

img = imread('homework.jpg');
fid = fopen('output/matlab_results.txt', 'w');

figimg = figure;
imshow(img);
hold all;

%% Load and plot lines

load('elements/s_points.mat', 'sx', 'sy');
load('elements/circumference.mat', 'a1', 'b1', 'c1', 'd1', 'e1', 'f1');
load('elements/lines.mat', 'l1', 'l2', 'l3', 'm1', 'm2' , 'm3', 'm4', 'm5', 'm6', 'h1', 'h2', 'h3', 'h4', 'lx');

cc1 = [a1, b1/2, d1/2; b1/2, c1, e1/2; d1/2, e1/2, f1];

% Compute important vertices to plot more easily
pointA = cross(l3, m5);
pointA = pointA/pointA(3);
pointB = cross(l3, m6);
pointB = pointB/pointB(3);
pointC = cross(l2, m6);
pointC = pointC/pointC(3);
pointD = cross(l2, m5);
pointD = pointD/pointD(3);
pointE = cross(l1, m1);
pointE = pointE/pointE(3);
pointF = cross(l1, m4);
pointF = pointF/pointF(3);
pointG = cross(lx, m1);
pointG = pointG/pointG(3);
pointH = cross(lx, m4);
pointH = pointH/pointH(3);

pointM = cross(lx, m2);
pointM = pointM/pointM(3);
pointN = cross(lx, m3);
pointN = pointN/pointN(3);
pointO = cross(l1, m2);
pointO = pointO/pointO(3);
pointP = cross(l1, m3);
pointP = pointP/pointP(3);

pointQ = cross(l2, h2);
pointQ = pointQ/pointQ(3);
pointR = cross(l2, h3);
pointR = pointR/pointR(3);

pointCV = pointC;
pointDV = pointD;
pointEV = pointE;
pointFV = pointF;

style = 'k--';

% Draw the conic
[cols, rows, ~] = size(img);
[x_grid, y_grid] = meshgrid(1:cols, 1:rows);

conic_values = a1*x_grid.^2 + b1*x_grid.*y_grid + c1*y_grid.^2 + d1*x_grid + e1*y_grid + f1;

hold on;
contour(x_grid, y_grid, conic_values, [0 0], style, 'LineWidth', 2);


% Interpolate the curve S
[x_fine, y_fine] = interpolateSpline(sx, sy);

figure(figimg)
title("Lines extracted")
%plot(sx, sy, 'o', 'MarkerSize', 8, 'DisplayName', 'Original Points'); % Original points
hold on;
plot(x_fine, y_fine, style, 'LineWidth', 2, 'DisplayName', 'Spline Interpolation'); % Spline curve


% Plot the lines
plot([pointA(1), pointB(1)], [pointA(2), pointB(2)], style, 'LineWidth', 2);
plot([pointB(1), pointC(1)], [pointB(2), pointC(2)], style, 'LineWidth', 2);
plot([pointC(1), pointD(1)], [pointC(2), pointD(2)], style, 'LineWidth', 2);
plot([pointD(1), pointA(1)], [pointD(2), pointA(2)], style, 'LineWidth', 2);
plot([pointE(1), pointF(1)], [pointE(2), pointF(2)], style, 'LineWidth', 2);
plot([pointE(1), pointG(1)], [pointE(2), pointG(2)], style, 'LineWidth', 2);
plot([pointF(1), pointH(1)], [pointF(2), pointH(2)], style, 'LineWidth', 2);

plot([pointE(1), pointD(1)], [pointE(2), pointD(2)], style, 'LineWidth', 2);
plot([pointF(1), pointC(1)], [pointF(2), pointC(2)], style, 'LineWidth', 2);

plot([pointM(1), pointO(1)], [pointM(2), pointO(2)], style, 'LineWidth', 2);
plot([pointN(1), pointP(1)], [pointN(2), pointP(2)], style, 'LineWidth', 2);

plot([pointO(1), pointQ(1)], [pointO(2), pointQ(2)], style, 'LineWidth', 2);
plot([pointP(1), pointR(1)], [pointP(2), pointR(2)], style, 'LineWidth', 2);

% Plot the names of the points
color = 'blue';

text(pointA(1), pointA(2), 'A', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointB(1), pointB(2), 'B', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointC(1), pointC(2), 'C', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointD(1), pointD(2), 'D', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointE(1), pointE(2), 'E', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointF(1), pointF(2), 'F', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointG(1), pointG(2), 'G', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointH(1), pointH(2), 'H', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointM(1), pointM(2), 'M', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointN(1), pointN(2), 'N', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointO(1), pointO(2), 'O', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointP(1), pointP(2), 'P', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointQ(1), pointQ(2), 'Q', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointR(1), pointR(2), 'R', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');

saveas(figimg, 'output/img_with_extracted_lines.png');
%% Problem 1

% Compute the vanishing points of the horizontal plane
vpl = cross(l2, l1);
vpm = cross(m5, m6);

fprintf(fid, 'The vanishing line is: \n');
vl = cross(vpl, vpm);
vl = vl/vl(3);
fprintf(fid, '[%.4f, %.4f, %.4f]\n\n', vl(1), vl(2), vl(3));

vpl = vpl/vpl(3);
vpm = vpm/vpm(3);


figure(figimg);
hold on;
plot(vpl(1), vpl(2), '.', 'Color', 'magenta', 'MarkerSize', 30);
plot(vpm(1), vpm(2), '.', 'Color', 'magenta', 'MarkerSize', 30);

plot([vpl(1), vpm(1)], [vpl(2), vpm(2)], 'Color', 'green', 'LineWidth', 2);

text(vpl(1), vpl(2), 'vpl', 'Color', 'black', 'FontSize', 12, 'FontWeight', 'bold');
text(vpm(1), vpm(2), 'vpm', 'Color', 'black', 'FontSize', 12, 'FontWeight', 'bold');

% plot([pointA(1), vpm(1)], [pointA(2), vpm(2)], 'Color', '#cccccc', 'LineWidth', 1, 'LineStyle', '-.');

% % Affine rectification (not requested, used just to check the results)
% H_a = [1, 0, 0; 0, 1, 0; vl(1), vl(2), vl(3)];
% tform = projective2d(H_a');
% R_out = imref2d(size(img)*4);
% rectified_img = imwarp(img, tform, 'OutputView', R_out);

% % Plot the rectified image
% figaff = figure;
% imshow(rectified_img);
% title('Affinely rectified Image');

%% Problem 2

% intersection of cc1 and vl
syms x y

eq1 = a1*x^2 + b1*x*y + c1*y^2 + d1*x + e1*y + f1 == 0;
eq2 = vl(1)*x + vl(2)*y + vl(3) == 0;

sol = solve([eq1, eq2], [x, y], 'IgnoreAnalyticConstraints',true,'Maxdegree',4);

imI = [double(sol.x(1)), double(sol.y(1)), 1];
imJ = [double(sol.x(2)), double(sol.y(2)), 1];

imI = imI'; % column vector
imJ = imJ'; % column vector

% Image of de Dual Conic of to the Circular Points
imDCCP = imI * imJ.' + imJ * imI.';
imDCCP = imDCCP/imDCCP(3,3);

clear x y

% Now, we can find the rectification matrix via SVD
[U, D, V] = svd(imDCCP);

D(3,3) = 1;

AAA = U * sqrt(D); % Numerical instability

% H_r is the homography scene --> world
H_r = double(inv(AAA));

pointA = H_r * pointA;
pointB = H_r * pointB;
pointC = H_r * pointC;
pointD = H_r * pointD;
pointA = pointA/pointA(3);
pointB = pointB/pointB(3);
pointC = pointC/pointC(3);
pointD = pointD/pointD(3);
pointE = H_r * pointE;
pointF = H_r * pointF;
pointG = H_r * pointG;
pointH = H_r * pointH;
pointE = pointE/pointE(3);
pointF = pointF/pointF(3);
pointG = pointG/pointG(3);
pointH = pointH/pointH(3);

% Finding the depth m
l = sqrt((pointA(1) - pointB(1))^2 + (pointA(2) - pointB(2))^2);
ratio = 1/l;

m = ratio * sqrt((pointA(1) - pointD(1))^2 + (pointA(2) - pointD(2))^2);
fprintf(fid, 'The depth m is equal to: %.4f\n\n', m);

% plot the rectified rectangle EFGH, the upper rectangle
figrecthor = figure;
axis equal;
hold on;
grid on;

% Plot the rectangles to verify that they are actually rectangles (i.e., the rectification may be correct)

% plot([pointA(1), pointB(1)], [pointA(2), pointB(2)], 'Color', 'red', 'LineWidth', 2);
% plot([pointB(1), pointC(1)], [pointB(2), pointC(2)], 'Color', 'red', 'LineWidth', 2);
% plot([pointC(1), pointD(1)], [pointC(2), pointD(2)], 'Color', 'red', 'LineWidth', 2);
% plot([pointD(1), pointA(1)], [pointD(2), pointA(2)], 'Color', 'red', 'LineWidth', 2);
plot([pointE(1), pointF(1)], [pointE(2), pointF(2)], 'Color', 'red', 'LineWidth', 2);
plot([pointE(1), pointG(1)], [pointE(2), pointG(2)], 'Color', 'red', 'LineWidth', 2);
plot([pointF(1), pointH(1)], [pointF(2), pointH(2)], 'Color', 'red', 'LineWidth', 2);
plot([pointH(1), pointG(1)], [pointH(2), pointG(2)], 'Color', 'red', 'LineWidth', 2);

% text(pointA(1), pointA(2), 'A', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
% text(pointB(1), pointB(2), 'B', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
% text(pointC(1), pointC(2), 'C', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
% text(pointD(1), pointD(2), 'D', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointE(1), pointE(2), 'E', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointF(1), pointF(2), 'F', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointG(1), pointG(2), 'G', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointH(1), pointH(2), 'H', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');

title("EFGH rectangle rectified as question 2")
saveas(figrecthor, 'output/rectified_EFGH.png');

% Computing the distance of O, P from E (this will be useful for the very last question)
pointO = H_r * pointO;
pointP = H_r * pointP;
pointO = pointO/pointO(3);
pointP = pointP/pointP(3);

EO_pxls = sqrt((pointE(1) - pointO(1))^2 + (pointE(2) - pointO(2))^2);
EP_pxls = sqrt((pointE(1) - pointP(1))^2 + (pointE(2) - pointP(2))^2);
EF_pxls = sqrt((pointE(1) - pointF(1))^2 + (pointE(2) - pointF(2))^2); % Which is 1 in the world

EO_real = EO_pxls / EF_pxls;
EP_real = EP_pxls / EF_pxls;

tform = projective2d(H_r');
rectified_img = imwarp(img, tform);
rect_fig_img = figure;
imshow(rectified_img);
title('Rectified Image');

saveas(rect_fig_img, 'output/rectified_img.png');


%% Problem 3

% Compute and plot the vanishing points of the vertical plane
vph = cross(h1, h4);
vph = vph/vph(3);

figure(figimg);
plot(vph(1), vph(2), '.', 'Color', 'magenta', 'MarkerSize', 30);
text(vph(1), vph(2), 'vph', 'Color', 'black', 'FontSize', 12, 'FontWeight', 'bold');

% H is the homography world --> scene
H = inv(H_r);

% Compute the intrinsic parameters of the camera
syms fx fy u0 v0

K = [fx 0 u0; 0 fy v0; 0 0 1];
omega = inv(K * K');

eqns = [
    H(:, 1)' * omega * H(:, 2) == 0;
    H(:, 1)' * omega * H(:, 1) - H(:, 2)' * omega * H(:, 2) == 0;
    vph' * omega * H(:, 1) == 0;
    vph' * omega * H(:, 2) == 0;
    ];

sol = solve(eqns, [fx, fy, u0, v0]);

K = double(abs([sol.fx 0 sol.u0; 0 sol.fy sol.v0; 0 0 1]));
fprintf(fid, 'The calibration matrix K is: \n');
fprintf(fid, '[%.4f, %.4f, %.4f;\n', K(1,1), K(1,2), K(1,3));
fprintf(fid, ' %.4f, %.4f, %.4f;\n', K(2,1), K(2,2), K(2,3));
fprintf(fid, ' %.4f, %.4f, %.4f]\n\n', K(3,1), K(3,2), K(3,3));

%% Problem 4

% Compute the elements of the first two columns of the homography of the vertical plane
syms h11 h12 h13 h21 h22

omega = inv(double(K)*double(K)');

h1 = [h11; h12; h13];
h2 = [h21; h22; 1];

equazioni = [
    h1' * omega * h2 == 0;
    h1' * omega * h1 - h2' * omega * h2 == 0;
    vpm' * omega * h1 == 0;
    vpm' * omega * h2 == 0;
    h1' * h2 == 0; % they are orthogonal
    ];

solh = solve(equazioni, [h11, h12, h13, h21, h22]);

h1 = [double(solh.h11); double(solh.h12); double(solh.h13)];
h2 = [double(solh.h21); double(solh.h22); 1];

HH = [solh.h11, solh.h12, solh.h13; solh.h21, solh.h22, 1; 0, 0, 0]';

% Compute the third column of the homography of the vertical plane, that is orthogonal to the first two
HH(:, 3) = cross(h1,h2);
HH = HH / HH(3,3);
HH = double(HH);

HH_r = inv(HH);


% Sanity check
pointCV = HH_r * pointCV;
pointDV = HH_r * pointDV;
pointCV = pointCV/pointCV(3);
pointDV = pointDV/pointDV(3);

pointEV = HH_r * pointEV;
pointFV = HH_r * pointFV;
pointEV = pointEV/pointEV(3);
pointFV = pointFV/pointFV(3);

figrectver = figure;
axis equal;
hold on;
grid on;

plot([pointEV(1), pointFV(1)], [pointEV(2), pointFV(2)], 'Color', 'red', 'LineWidth', 2);
plot([pointCV(1), pointDV(1)], [pointCV(2), pointDV(2)], 'Color', 'red', 'LineWidth', 2);
plot([pointCV(1), pointFV(1)], [pointCV(2), pointFV(2)], 'Color', 'red', 'LineWidth', 2);
plot([pointDV(1), pointEV(1)], [pointDV(2), pointEV(2)], 'Color', 'red', 'LineWidth', 2);

text(pointCV(1), pointCV(2), 'C', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointDV(1), pointDV(2), 'D', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointEV(1), pointEV(2), 'E', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');
text(pointFV(1), pointFV(2), 'F', 'Color', color, 'FontSize', 12, 'FontWeight', 'bold');


% Compute the height h, same as in Problem 2
ratio = 1/sqrt((pointCV(1) - pointDV(1))^2 + (pointCV(2) - pointDV(2))^2);

h = ratio * sqrt((pointCV(1) - pointFV(1))^2 + (pointCV(2) - pointFV(2))^2);
fprintf(fid, 'The height h is equal to: %.4f\n\n', h);

save('elements/real_distances.mat', 'm', 'h', 'EO_real', 'EP_real');

title("CDEF rectangle rectified as question 4")
saveas(figrectver, 'output/rectified_CDEF.png');

%% Problem 5

fprintf(fid, 'XY localization of points of S:\n');
fprintf(fid, '[x_image, y_image;\n');
fprintf(fid, 'x_world, y_world]\n\n');

% Rectify the curve
for i = 1:length(sx)
    point = [sx(i); sy(i); 1];
    point = H_r * point;
    point = point/point(3);
    
    fprintf(fid, '[%.4f, %.4f;\n %.4f, %.4f]\n\n', sx(i), sy(i), point(1), point(2));
    
    sx(i) = point(1);
    sy(i) = point(2);
end
fprintf(fid, '\n');

% % Plot original data and interpolated curve
% figure(figrecthor);
% plot(sx, sy, 'o', 'MarkerSize', 8, 'DisplayName', 'Original Points'); % Original points
% hold on;
% plot(x_fine, y_fine, 'LineWidth', 2, 'DisplayName', 'Spline Interpolation'); % Spline curve

saveas(figrecthor, 'output/rectified_curve_with_horiz_plane.png');

save('elements/s_points_rect.mat', 'sx', 'sy');

%% Problem 6

fprintf(fid, 'The localization of the parallelogram vertices in the camera reference system is:\n\n');

% H = K * [rpx, rpy, op]
ro = K \ H;

rpx = ro(:, 1);
fprintf(fid, 'rpx = [%.4f; %.4f; %.4f]\n\n', rpx(1), rpx(2), rpx(3));

rpy = ro(:, 2);
fprintf(fid, 'rpy = [%.4f; %.4f; %.4f]\n\n', rpy(1), rpy(2), rpy(3));

% rpz is orthogonal to rpx and rpy
rpz = cross(rpx, rpy);
fprintf(fid, 'rpz = [%.4f; %.4f; %.4f]\n\n', rpz(1), rpz(2), rpz(3));

op = ro(:, 3);
fprintf(fid, 'op = [%.4f; %.4f; %.4f]\n\n', op(1), op(2), op(3));

% Show the homographies we found

fprintf(fid, '--------------------------------------------------------\n');
fprintf(fid, 'The homography world->image for the horizontal plane is:\n');
fprintf(fid, '[%.4f,\t%.4f,\t%.4f;\n', H(1,1) , H(1,2), H(1,3));
fprintf(fid, ' %.4f,\t%.4f,\t%.4f;\n', H(2,1) , H(2,2), H(2,3));
fprintf(fid, ' %.4f,\t%.4f,\t%.4f]\n\n', H(3,1) , H(3,2), H(3,3));

fprintf(fid, 'The homography world->image for the vertical plane is:\n');
fprintf(fid, '[%.4f,\t%.4f,\t\t%.4f;\n', HH(1,1) , HH(1,2), HH(1,3));
fprintf(fid, ' %.4f,\t%.4f,\t\t%.4f;\n', HH(2,1) , HH(2,2), HH(2,3));
fprintf(fid, ' %.4f,\t%.4f,\t\t%.4f]\n\n', HH(3,1) , HH(3,2), HH(3,3));


% Close the output file
fclose(fid);
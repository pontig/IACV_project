function [x_fine, y_fine] = interpolateSpline(sx, sy)
    % Generate the curve S via spline interpolation
    t = 1:length(sx);  % Arbitrary parameterization (can be customized)
    % Create splines for x(t) and y(t)
    spline_x = spline(t, sx); % Spline for x as a function of t
    spline_y = spline(t, sy); % Spline for y as a function of t
    % Define a finer set of t values for interpolation
    t_fine = linspace(min(t), max(t), 100);
    % Evaluate the splines at the finer t values
    x_fine = ppval(spline_x, t_fine);
    y_fine = ppval(spline_y, t_fine);
end
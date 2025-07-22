function [] = plotLine(l, color)
    if nargin < 2
        color = 'r';
    end
    x = linspace(-10, 10, 100);
    y = (-l(1) * x - l(3)) / l(2);
    plot(x, y, color);
    xlim([-10 10]);
    ylim([-10 10]);
    grid on;
end
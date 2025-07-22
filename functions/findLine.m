function l = findLine(i, j, pts)
    % Find line that passes through points i and j
    pt1 = pts(i, :);
    pt2 = pts(j, :);

    l = cross(pt1, pt2);
    l = (l / l(3))';
end
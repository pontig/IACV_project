function [l] = segToLine(pts)
    a = [pts(1,:)';1]';
    b = [pts(2,:)';1]';
    l = cross(a,b);
    l = l./l(3);
end
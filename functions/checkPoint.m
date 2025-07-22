% Function to check if point satisfies equations
function err = checkPoint(x, y, a1, b1, c1, d1, e1, f1, vl)
    err1 = abs(a1*x^2 + b1*x*y + c1*y^2 + d1*x + e1*y + f1);
    err2 = abs(vl(1)*x + vl(2)*y + vl(3));
    err = [err1; err2];
end
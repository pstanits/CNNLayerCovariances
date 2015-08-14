function lgm = my_logm( C1 )
% Custom Matrix logarithm
    [u,e] = schur(C1);
    e = diag(e);
    e(e <= 10^(-5)) = 1;
    inter = log(diag(e));
    inter(isinf(inter)) = 0;
    lgm = u * inter * u';
end


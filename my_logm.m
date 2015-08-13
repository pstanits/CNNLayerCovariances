function lgm = my_logm( C1 )
% Custom Matrix logarithm
[u,e] = schur(C1);
e = diag(e);
e(e<=1e-5) = 1;
lgm = u * log(diag(e)) * u';
end


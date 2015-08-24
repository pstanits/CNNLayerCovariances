% usage: for two symmetric postiive definite matrices A, B, d = jbld(A,B) 
% computes the Jensen-Bregman Logdet Divergence between A and B.
% for questions, please email: cherian@cs.umn.edu
function d = jbld(A,B)
n=size(A,1);
if n < 13
    d = log(det(A+B)) - 0.5*log(det(A)) - 0.5*log(det(B)) - n*log(2);
else
    if rank(A) < size(A,1)
        A = A + 2 * 10^(-6) * eye(size(A,1));
    end
    if rank(B) < size(B,1)
        B = B + 2 * 10^(-6) * eye(size(B,1));
    end
    
    r = chol(A+B);
    nr=2*sum(log(diag(r)));
    
    r1=chol(A);
    dr1=sum(log(diag(r1)));
    
    r2=chol(B);
    dr2=sum(log(diag(r2)));

    d = nr - dr1 - dr2 - n*log(2);
end
end
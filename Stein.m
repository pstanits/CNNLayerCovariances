function distance = Stein( C1,C2 )
% Log Euclidean Distance between matrices
% Panagiotis Stanitsas
% University of Minnesota
distance = sqrt(log(det(C1 + C2) / 2) - 0.5 * log(det(C1 * C2)));
end
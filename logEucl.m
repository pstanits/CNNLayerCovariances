function distance = logEucl( C1,C2 )
% Log Euclidean Distance between matrices
% Panagiotis Stanitsas
% University of Minnesota
distance = norm((logm(C1) - logm(C2)),'fro');
end


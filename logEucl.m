function distance = logEucl( C1,C2 )
% Log Euclidean Distance between matrices
% Panagiotis Stanitsas
% University of Minnesota
distance = norm((log(C1) - log(C2)),'fro');
end


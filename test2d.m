clear;

data=nicering;
[som, grid]=som2d(data, 10, 10, 30000, 0.5, 5);
% Quantization error is the average distance between each data vector and its BMU.
qe = quantization_error(som, data);
som2d_vis(som, grid, data);
title(['QE = ',num2str(qe)])
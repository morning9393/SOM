clear;

data=nicering;
som=som1d(data, 20, 5000, 0.5, 10);
% Quantization error is the average distance between each data vector and its BMU.
qe = quantization_error(som, data);
som1d_vis(som, data);
title(['QE = ',num2str(qe)])

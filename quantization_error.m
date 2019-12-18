function qe = quantization_error(som, data)

dataSize = size(data, 1);
somSize = size(som, 1);

total_d = 0;
for i=1:dataSize
    sample = data(i,:);
    diff = som - repmat(sample, somSize, 1);
    d = sqrt(sum(diff .* diff, 2));
    total_d = total_d + min(d);
end
qe = total_d / dataSize;
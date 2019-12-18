function som = som1d (trainingData, neuronCount, trainingSteps, startLearningRate, startRadius)
% som = lab_som (trainingData, neuronCount, trainingSteps, startLearningRate, startRadius)
% -- Trains a 1D SOM i.e. A SOM where the neurons are arranged
%             in a single line. 
%             
% -- <trainingData> data to train the SOM with
% -- <som> returns the neuron weights after training
% -- <neuronCount> number of neurons 
% -- <trainingSteps> number of training steps 
% -- <startLearningRate> initial learning rate
% -- <startRadius> initial radius used to specify the initial neighbourhood size

% Initialize the constant parameters of learn rate decay rule
% and neighborhood size decay rule.
t1 = trainingSteps / 4;
t2 = trainingSteps / 3;

% Randomly select initial neurons' weights from data set.
dataSize = size(trainingData, 1);
som = trainingData(randperm(dataSize, neuronCount), :);

% Learning begin.
for t = 0:trainingSteps - 1
    % Ramdomly select a sample from data set.
    sample = trainingData(randperm(dataSize, 1), :);
    
    % Calaulate the Euclidean distance from each neuron to the sample and 
    % get the position of BMU, who has the minimum distance.
    diff = som - repmat(sample, neuronCount, 1);
    norms = sum(diff .* diff, 2);
    [~, win_p] = min(norms);
    
    % Calculate the learn rate decay rule at this step.
    learningRate = startLearningRate * exp(-1 * t / t1);
    % Calculate the neighborhood size decay rule at this step.
    radius = startRadius * exp(-1 * t / t2);
    
    % Traversing all neurons to update the weights of BMU and its neighbors.
    for i=1:neuronCount
        % Calculate the lattice distance from a neurons to BMU by their 
        % absolute value of coordinate difference
        d = abs(i - win_p);
        if d <= radius 
            % Calculate the new weights of BMU or its neighbors.
            h = exp(-1 * d^2 / (2 * radius^2));
            som(i, :) = som(i, :) + learningRate * h * (sample - som(i, :));
        end
    end
end


function [som,grid] = som2d (trainingData, neuronCountW, neuronCountH, trainingSteps, startLearningRate, startRadius)
% som = lab_som2d (trainingData, neuronCountW, neuronCountH, trainingSteps, startLearningRate, startRadius)
% -- Trains a 2D SOM, which consists of a grid of
%             (neuronCountH * neuronCountW) neurons.
%             
% -- <trainingData> data to train the SOM with
% -- <som> returns the neuron weights after training
% -- <grid> returns the location of the neurons in the grid
% -- <neuronCountW> number of neurons along width
% -- <neuronCountH> number of neurons along height
% -- <trainingSteps> number of training steps 
% -- <startLearningRate> initial learning rate
% -- <startRadius> initial radius used to specify the initial neighbourhood size
%

% Function will still return the a weight matrix 'som' with
% the same format as described in lab_som().
%
% However, it will additionally return a vector 'grid' that will
% state where each neuron is located in the 2D SOM grid. 
% 
% grid(n, :) contains the grid location of neuron 'n'
%
% For example, if grid = [[1,1];[1,2];[2,1];[2,2]] then:
% 
%   - som(1,:) are the weights for the neuron at position x=1,y=1 in the grid
%   - som(2,:) are the weights for the neuron at position x=2,y=1 in the grid
%   - som(3,:) are the weights for the neuron at position x=1,y=2 in the grid 
%   - som(4,:) are the weights for the neuron at position x=2,y=2 in the grid
%


% Initialize the constant parameters of learn rate decay rule
% and neighborhood size decay rule.
t1 = trainingSteps / 4;
t2 = trainingSteps / 3;

% Initialize the neuron grid and indicate their positions.
neuronCount = neuronCountW * neuronCountH;
grid = zeros(neuronCount, 2);
for i = 1:neuronCountH
    for j = 1:neuronCountW
        grid(neuronCountW * (i - 1) + j,:) = [j, i];
    end
end

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
    [~, winner] = min(norms);
    p_win = grid(winner, :);
    
    % Calculate the learn rate decay rule at this step.
    learningRate = startLearningRate * exp(-1 * t / t1);
    % Calculate the neighborhood size decay rule at this step.
    radius = startRadius * exp(-1 * t / t2);
    
    % Traversing all neurons to update the weights of BMU and its neighbors.
    for i=1:neuronCount
        % Calculate the lattice distance from a neurons to BMU by their 
        % sum of absolute value of coordinate differences
        p = grid(i, :);
        d = abs(p_win(1) - p(1)) + abs(p_win(2) - p(2));
        if d <= radius 
            % Calculate the new weights of BMU or its neighbors.
            h = exp(-1 * d^2 / (2 * radius^2));
            som(i, :) = som(i, :) + learningRate * h * (sample - som(i, :));
        end
    end
end


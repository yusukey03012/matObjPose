function net = insertBNLayers(net)
%INSERTBNLAYERS adds batch normalization to a network
%  INSERTBNLAYERS(dagnet) inserts batch normalization
%  layers directly after each convolutional layer in the
%  the a DagNN network

% return if the network already contains batch norm layers 
for l = 1:numel(net.layers)
    if isa(net.layers(l).block, 'dagnn.BatchNorm')
        return;
    end
end

% loop over the network and insert batch norm layers after 
% convolutions
layerOrder = net.getLayerExecutionOrder();
for l = layerOrder
    if isa(net.layers(l).block, 'dagnn.Conv') 
        if  ~strcmp(net.layers(l).outputs, 'prediction') && ~strcmp(net.layers(l).outputs, 'prediction1')&&~strcmp(net.layers(l).outputs, 'prediction2')  
        net = addBatchNorm(net, l);
        end
    end
end

net.rebuild()


function net = addBatchNorm(net, layerIndex)
%ADDBATCHNORM adds a batch norm layer
%    ADDBATCHNORM adds a batch layer to the network
%    to a network at the index given by `layerIndex`

% pair inputs and outputs to ensure a valid network
inputs = net.layers(layerIndex).outputs;

% find the number of channels produced by the previous layer
numChannels = net.layers(layerIndex).block.size(4);

outputs = sprintf('xbn%d',layerIndex);

% Define the name and parameters for the new layer
name = sprintf('bn%d', layerIndex);

block = dagnn.BatchNorm();
paramNames = {sprintf('%sm', name) ...
              sprintf('%sb', name) ...
              sprintf('%sx', name) };

% add new layer to the network          
net.addLayer(...
    name, ...
    block, ...
    inputs, ...
    outputs, ...
    paramNames) ;


% set mu (gain parameter)
mIdx = net.getParamIndex(paramNames{1});
net.params(mIdx).value = ones(numChannels, 1, 'single');
net.params(mIdx).learningRate = 2;
net.params(mIdx).weightDecay = 0;

% set beta (bias parameter)
bIdx = net.getParamIndex(paramNames{2});
net.params(bIdx).value = zeros(numChannels, 1, 'single');
net.params(bIdx).learningRate = 1;
net.params(bIdx).weightDecay = 0;

% set moments parameter
xIdx = net.getParamIndex(paramNames{3});
net.params(xIdx).value = zeros(numChannels, 2, 'single');
net.params(xIdx).learningRate = 0.05;
net.params(xIdx).weightDecay = 0;

% modify the next layer to take the new inputs
net.layers(layerIndex + 1).inputs = {outputs};
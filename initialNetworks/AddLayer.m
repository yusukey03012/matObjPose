function net = AddLayer(net, layer, varargin)

global layerIndex
opts.batchNorm = 1;
opts.name = ['layer', num2str(layerIndex)];
opts.params = {};
opts.out = {['x', num2str(layerIndex)]};
opts.scale = 1;
        

if isempty(net.layers)
  opts.in= {};
else 
  opts.in  = net.layers(end).outputs;
end

opts = vl_argparse(opts, varargin);

if isa(layer, 'dagnn.Conv')
    opts.params = {['convf',num2str(layerIndex)], ['convb',num2str(layerIndex)]};
    
elseif isa(layer, 'dagnn.ConvTranspose')
   opts.params = {['deconvf' , num2str(layerIndex)]};
end

net.addLayer(opts.name,...
             layer,...
             opts.in,...
             opts.out,... % output
             opts.params); % params

layerIndex= layerIndex +1;


if isa(layer, 'dagnn.Conv')
    
        opts.weightInitMethod = 'xavierimproved';
        opts.weightInitMethod = 'gaussian';
        
        net.params(end-1).value =...
              init_weight(opts, layer.size(1), layer.size(2), layer.size(3), layer.size(4), 'single') ;
        net.params(end-1).learningRate=1;
        net.params(end-1).weightDecay=1;
                
        net.params(end).value = zeros(1, layer.size(4),'single');
        net.params(end).learningRate=2;
        net.params(end).weightDecay=0;
    
if opts.batchNorm 
    net = addBatchNorm(net, layerIndex);
    layerIndex= layerIndex +1;

end
elseif isa(layer, 'dagnn.ConvTranspose')
    net.params(end).value = ones(2,2,f,f,'single');
    net.params(end).learningRate = 0 ;
    net.params(end).weightDecay = 1 ;
end
end


function net = addBatchNorm(net, layerIndex)
%ADDBATCHNORM adds a batch norm layer
%    ADDBATCHNORM adds a batch layer to the network
%    to a network at the index given by `layerIndex`

% pair inputs and outputs to ensure a valid network
inputs = net.layers(layerIndex-1).outputs;

% find the number of channels produced by the previous layer
numChannels = net.layers(layerIndex-1).block.size(4);

outputs = sprintf('x%d',layerIndex);

% Define the name and parameters for the new layer
name = sprintf('layer%d', layerIndex);

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
net.params(mIdx).learningRate = 1;
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
%net.layers(layerIndex + 1).inputs = {outputs};
end
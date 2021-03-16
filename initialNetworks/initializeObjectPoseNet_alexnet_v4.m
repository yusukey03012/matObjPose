net = load('data/model/imagenet-caffe-alex.mat');

net = dagnn.DagNN.fromSimpleNN(net,'CanonicalNames',1);
net.removeLayer('prob')
net.removeLayer('fc8')

outAlex = net.layers(end).outputs{1};
net = AddLayer(net, dagnn.Conv('size',[1, 1, 4096, 9],'pad',0,'stride',1,'hasBias',true),'batchNorm', 0, 'in',{outAlex}, 'out', 'rotation', 'name', 'fc8');
net.params(end).learningRate=2;net.params(end).weightDecay=0;
net = AddLayer(net, dagnn.RegLoss('loss', 'l2loss'),'in', {'rotation', 'label'},'out', 'objective_angle', 'name', 'loss');
net.rebuild();   


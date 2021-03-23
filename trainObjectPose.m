function trainObjectPose(varargin)

opts.outNode=3;
opts.inNode=3;
opts.gpus = [3];
opts.expDir = [];
opts.imdbPath = [];
opts.numSubSample = 1;
opts.baseModel=[];

opts.datas=' ';
opts.aug=0;
opts.NoAug=1; %used for calling the correct imdb creation function

opts.batchSize = 40;
opts.numEpochs = 50;
opts.learningRate = 0.01*[0.01*ones(1, 20) 0.005*ones(1, 10) 0.001*ones(1, 10) 0.0005*ones(1, 10)] ;
opts.batchNormalization = 1;%useful for big networks

%GPU
opts.lossFunc='l2loss';

%export path, imdb store path and location of training / validation data.
opts.expDir = ' ';
opts.imdbPath = ' ';

%objectives
opts.derOutputs = {'objective1',1};

opts.averageImage = single(repmat(128,1,1,opts.inNode));
opts.imageSize = [248, 248] ;
opts.train.momentum=0.95;

opts.net=[];

%imdb generation function
opts.batchFun = 'getBatchDagNN';
opts = vl_argparse(opts, varargin);

opts.imdbfn =@getImdbNoAug;

opts = vl_argparse(opts, varargin);
run(opts.net)
opts.net=net;

%create imdb and train
cnn_regressor_dag(opts);

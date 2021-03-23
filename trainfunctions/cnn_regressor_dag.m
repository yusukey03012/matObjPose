function [net, info] = cnn_regressor_dag(varargin)

%Create the imdb and train the model

% Augmentation settings
opts.aug=0;
opts.NoAug=0;
opts.baseModel = [];
% Export directory for model and imdb
opts.expDir = ' ';
opts.imdbPath = ' ';
opts.train.batchSize = 256 ;
opts.train.numSubBatches = 1;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.derOutputs= {'objective', 1} ;
opts.train.learningRate = [0.001*ones(1, 17) 0.0005*ones(1, 50) 0.002*ones(1, 500)  0.03*ones(1, 130) 0.01*ones(1, 100)] ;
opts.train.momentum=0.9;
opts.useBnorm = false ;
opts.batchNormalization = 0;
opts.train.prefetch = false ;

opts.train.train = [];
opts.train.val = [];

%GPU
opts.train.gpus = [];
opts.outNode=3;
opts.inNode=3;

% IMDB generation function
opts.imdbfn= [];

% Batch parameters
bopts.averageImage = single(repmat(128,1,1,opts.inNode));
bopts.imageSize = [120, 80] ;
bopts.batchFun= 'getBatchDagNN';

% Parse settings
[opts, trainParams] = vl_argparse(opts, varargin); %main settings
[opts.train, boptsParams]= vl_argparse(opts.train, trainParams); %train settings
[bopts, netParams]= vl_argparse(bopts, boptsParams); %batch settings
net=netParams{1}.net; %network
clear trainParams boptsParams netParams;

useGpu = numel(opts.train.gpus) > 0 ;
bopts.GPU=useGpu;
%Paths OSX / Ubuntu
opts.train.expDir = opts.expDir ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------
if exist(opts.imdbPath)
    imdb = load(opts.imdbPath);
else
    imdb = opts.imdbfn(opts);
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb','-v7.3') ;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
fn = getBatchDagNNWrapper(bopts,useGpu) ;
info = cnn_train_dag_reg(net, imdb, fn, opts.train) ;

function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
eval(['fn = @(imdb,batch,mode)', opts.batchFun, '(imdb, batch,mode, opts, useGpu)']);



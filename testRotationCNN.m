matconvnetpath = '..\Documents\MATLAB\matconvnet-1.0-beta25';

addpath('trainfunctions');
addpath('initialnetworks');
addpath('utils');
addpath('model-train');
addpath('external')

run(fullfile(matconvnetpath, 'matlab', 'vl_setupnn.m'))
%%

imdbPath = 'data\tool_yellow.mat'
imdb = load(imdbPath);

netPath = ['experiments\objectPose\tool1\net-epoch-15.mat'];% network path
%netPath = ['data\model\net-epoch-15.mat'];

load(netPath)
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';     % test mode
net.conserveMemory=0;  % output intermediate results or not
%net.move('gpu')        % uncommnet if you use gpu
%% remove loss layer
for i = 1
    net.removeLayer(net.layers(end).name)
end
%%


file1 = ['data\tool_yellow_no_col.off'];
[vertex1,face1] = read_off(file1);
mesh1.vertex = vertex1;
mesh1.face = face1;
mesh1.vertex = bsxfun(@minus, mesh1.vertex, mean(mesh1.vertex));
scale = max(max(mesh1.vertex) - min(mesh1.vertex));
mesh1.vertex = scale * mesh1.vertex;


%%
close all
figure;
subplot(1,2,2)
h=my_patch(mesh1.vertex, mesh1.face);
set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
set(gca, 'Projection', 'perspective');
view(0,0)
camlight('infinite')
xlabel('x');ylabel('y');zlabel('z')
axis equal;axis auto;axis equal;axis off  
%%
testset = find(imdb.images.set==2);
for i= 1:length(testset)
    i
    
    batch = testset(i);
    opts.imageSize = [227, 227];
    opts.averageImage = single(repmat(128,1,1, 3));
    offset = opts.averageImage ;
    im= imdb.images.data{1,batch};
    im_ = single(im);
    im_ = bsxfun(@minus, im_, offset) ; % subtract average 
    im_ = im_./256; % simple normalization
    im_ = imresize(im_, opts.imageSize);
    %im_= gpuArray(im_); %uncommnet if you use gpu
    inputs = {'input', im_};
    
    net.eval(inputs) ; % forward pass
    
    R = double(reshape(gather(net.vars(net.getVarIndex('rotation')).value),[3,3])); % result
    R= PolarDecomposition(R, ' ');

    subplot(1,2,1)
    image(im)
    axis equal
    axis off       
    subplot(1,2,2)
    h.Vertices = (R * mesh1.vertex')';
    
    pause(0.5)   

    
end

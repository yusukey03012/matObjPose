clear
addpath('trainfunctions');
addpath('initialnetworks');
addpath('model-train');
addpath('utils');
run(fullfile('C:\Users\yusukey0301\Documents\MATLAB\matconvnet-1.0-beta25', 'matlab', 'vl_setupnn.m'))
%%
imdbPath = 'data\lipton.mat'
imdb = load(imdbPath);
% network dir
netPath = ['data\model\net-epoch-15.mat'];
netPath = ['experiments\objectPose\lipton1\net-epoch-15.mat'];

load(netPath)
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';     % test mode
net.conserveMemory=0;  % output intermediate results or not
%net.move('gpu')        % to gpu
%%

file1 = ['data\milk_carton.off'];
[vertex1,face1] = read_off(file1);
mesh1.vertex = vertex1;
mesh1.face = face1;
mesh1.vertex = bsxfun(@minus, mesh1.vertex, mean(mesh1.vertex));
scale = max(max(mesh1.vertex) - min(mesh1.vertex));
mesh1.vertex = scale * mesh1.vertex;

%%
for i = 1
    net.removeLayer(net.layers(end).name)
end
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
    im_ = bsxfun(@minus, im_, offset) ; % •½‹Ï’l?i256/2?j‚ðˆø‚­
    im_ = im_./256; % normalize
    im_ = imresize(im_, opts.imageSize);
    %inputs = {'input', gpuArray(im_)};
    inputs = {'input', im_};
    
    net.eval(inputs) ; % forward pass
    
    %aa = [ 0.82391286; -0.03452247;  0.26206875];
    %aa = double(reshape(gather(net.vars(net.getVarIndex('axisAngle')).value),[3,1])) % result
        
    R = double(reshape(gather(net.vars(net.getVarIndex('rotation')).value),[3,3])); % result
    R= PolarDecomposition(R, ' ');
    %aa_correct= imdb.images.labels{1,batch};
    %pause
    %aa= Rot2Axis(R); % axis angle -> rotation matrix
    
    %R_err = R * R_correct' ;
    %aa_err= Rot2Axis(R_err); % axis angle -> rotation matrix
    %error = norm(aa_err)* 180/pi
    %error = norm(aa -aa_correct)* 180/pi;
    %error_sum = error_sum + error;
    %R = Axis2Rot(aa); % axis angle -> rotation matrix
    %R_correct = Axis2Rot(aa_correct)
    
    %     class = double(reshape(gather(net.vars(net.getVarIndex('x32')).value),[3,1])); % result
    %     [prediction,object_class] = max(class);
    %
    subplot(1,2,1)
    image(im)
    axis equal
    axis off       
    subplot(1,2,2)
    h.Vertices = (R * mesh1.vertex')';
    
    pause(0.1)   
    % close all
    
end
%error_avg = error_sum/length(imdb.images.data)
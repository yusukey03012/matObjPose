clear 

addpath('utils')
%%

depth_thresh = 3;

file = 'data\pc000000_.pcd';

[data, I, rgb, xyd] = readPCD(file, 1,  depth_thresh);

[v,f] =read_off('data\lipton_rot.off');
mesh = Mesh(v/1000, f, 'simple');

bbox = [340,95,497,227];

xyd_extract = xyd(bbox(2):bbox(4), bbox(1):bbox(3),:);
xyd_extract2 = nan(size(xyd));
xyd_extract2(bbox(2):bbox(4), bbox(1):bbox(3),:) = xyd_extract;
x = xyd_extract2(:,:,1);
y = xyd_extract2(:,:,2);
z = xyd_extract2(:,:,3);
x(x==0) = nan;
y(y==0) = nan;
z(z==0) = nan;

%x_fill = fillHoles(x, 'recursive-dilate');
%y_fill = fillHoles(y, 'recursive-dilate');
%z_fill = fillHoles(z, 'recursive-dilate');
%vertex_extract = [x_fill(:),y_fill(:),z_fill(:)];
%xyd_extract_fill = cat(3,x_fill,y_fill,z_fill);
%xyd_center = reshape(xyd_extract_fill(round(size(xyd_extract_fill,1)/2) ,round(size(xyd_extract_fill,1)/2),:), [1,3]);

sampleRate = 1;
[Nx,Ny,Nz] = surfnorm(x,y,z);
xyz = [x(:),y(:),z(:)];
N = [Nx(:),Ny(:),Nz(:)];
idx = find(~isnan(xyz(:,1))&~isnan(N(:,1)));
idx2 = find(xyz(idx,3) < depth_thresh);
xyz_subsample = xyz(idx(idx2(1:sampleRate:end)),:);
N_subsample = N(idx(idx2(1:sampleRate:end)),:);
data_extract = PointNormal(xyz_subsample, [], -N_subsample);
xyd_center = median(data_extract.vertex)

R_init =  Rotation_by_Axis('y', 90) * Rotation_by_Axis('x', -90);
t_init = - mean(mesh.vertex) + xyd_center;
p2=(( R_init * mesh.vertex'))';
pTranslate = bsxfun(@plus, bsxfun(@minus, p2, mean(p2) ), xyd_center);
v_init =pTranslate;
model = Mesh(v_init, mesh.face, 'simple');
model.ComputeNormal()
%%
isVis =1
if isVis ==1
    close all
    figure('color', [0.9, 0.9, 0.9])
    h = my_patch(model.vertex,model.face);
    hold on
    plot3(data_extract.vertex(:,1), data_extract.vertex(:,2), data_extract.vertex(:,3), '.')
    xlabel('x');ylabel('y');zlabel('z')
    view(150,300)
    camlight('infinite')
    
end

tic
matcher = ClosestPointMatcher(model, data_extract);
option.dst = 0.1; 
option.theta =180;
option.project = 0; 
option.visibility=1; 
option.viewpoint = [0,0,0];

% optimize translation
for iter = 1:30
    iter
    
    %[idxS, idxT, goalPos]= matcher.ComputeTargetToSourceMatch(option);
    [idxS, idxT, goalPos]= matcher.ComputeSourceToTargetMatch(option);
    [T, r, s, t1, t2,k] = ComputeGlobalTransformation(model.vertex, data_extract.vertex, idxS,idxT);
    r = eye(3);
    t= (r* (-t1)')' +  t2;
    model.vertex = (r * (model.vertex)')' + repmat(t, length(model.vertex),1);
    model.vNormal=  (r *(model.vNormal)')';
    
    if isVis ==1
        h.Vertices = model.vertex ;
        drawnow
    end
end
toc;


tic
% optimize rotation + translation
option.dst = 0.1; 
option.theta =180;
option.project = 0; 
option.visibility= 0; 
option.viewpoint = [0,0,0];

for iter = 1:50
    
    iter
    [idxS, idxT, goalPos]= matcher.ComputeTargetToSourceMatch(option);
    [T, r, s, t1, t2,k] = ComputeGlobalTransformation(model.vertex, data_extract.vertex, idxS,idxT);
    
    Src = model.vertex(idxS,:);
    Dst = data_extract.vertex(idxT,:);
    Normals = data_extract.vNormal(idxT,:);
    
    [x]=minimize_point_to_plane(Src, Dst, Normals);
    M=get_transform_mat(x);
    
    r = M(1:3,1:3);
    t = M(1:3,4)';

    model.vertex = (r * (model.vertex)')' + repmat(t, length(model.vertex),1);
    model.vNormal=  (r *(model.vNormal)')';
    
    if isVis ==1
        h.Vertices = model.vertex ;
        drawnow
    end
end
toc
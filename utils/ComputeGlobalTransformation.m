function [T, R, s, t1, t2,k] = ComputeGlobalTransformation(model,target,idx_model,idx_target)

vec1=(model(idx_model,:) - repmat(mean(model(idx_model, :)), length(idx_model),1))';
vec2=(target(idx_target,:) - repmat(mean(target(idx_target,:)), length(idx_model) ,1))';

T = vec2/ vec1;
[u,k,v]=svd(T);
D = diag([1,1,det(u*v')]);
R=u*D*v';
R= u*v';
s = ((k(1,1).^2+ k(2,2).^2+k(3,3).^2)/3).^0.5;

t1 = mean( model(idx_model,:));
t2 = mean(target(idx_target,:));% - mean(source.vertex(idx,:));

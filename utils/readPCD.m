function [data, I, rgb,xyd] = readPCD(file, sampleRate,depth_thresh)

width = dlmread(file, ' ' , [6 1 6 1]);
height = dlmread(file, ' ' , [7 1 7 1]);

if width==1|height ==1
    width = 640;
    height = 480;
end

xyzc = dlmread(file,' ', 11,0);
rgba = uint32(xyzc(:,4));
rgb = unpackRGBFloat(rgba);


r = reshape(rgb(:,1), width, height)';
g = reshape(rgb(:,2), width, height)';
b = reshape(rgb(:,3), width, height)';

x = reshape(xyzc(:,1), width, height)';
y = reshape(xyzc(:,2), width, height)';
z = reshape(xyzc(:,3), width, height)';

xyd = cat(3,x,y,z);

[Nx,Ny,Nz] = surfnorm(x,y,z);

xyz = [x(:),y(:),z(:)];
N = [Nx(:),Ny(:),Nz(:)];
idx = find(~isnan(xyz(:,1)));
idx2 = find(xyz(idx,3) < depth_thresh);
xyz_subsample = xyz(idx(idx2(1:sampleRate:end)),:);
N_subsample = N(idx(idx2(1:sampleRate:end)),:);

data = PointNormal(xyz_subsample, [], -N_subsample);

I = zeros(height,width,3);
I(:,:,1) = r;
I(:,:,2) = g;
I(:,:,3) = b;
I= uint8(I);
r2 = r(idx(idx2(1:sampleRate:end)));
g2 = g(idx(idx2(1:sampleRate:end)));
b2 = b(idx(idx2(1:sampleRate:end)));

rgb = [r2,g2,b2];

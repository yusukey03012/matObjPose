function inputs = getBatch_augment(imdb, batch, mode, opts, useGpu)

im = 128* ones(opts.imageSize(1), opts.imageSize(2), 3, numel(batch), 'single') ;
offset = opts.averageImage ;

% bbox shift
for i= 1:2:length(batch)
    
    I = single(imdb.images.data{1,batch(i)});
       
    I = imresize(I, opts.imageSize);
    im(:,:,:,i) = bsxfun(@minus, I, offset);
    im(:,:,:,i) = im(:,:,:,i)./256;
    im_ = im(:,:,:,i);
    dx = randi(21) - 11 ;
    dy = randi(21) - 11 ;
    cx = 11:opts.imageSize(1)-11;
    cy= cx;
    sx = cx + dx;
    sy = cx + dy;
    im(sy,sx,:,i) = im_(cx,cy,:);
    
end

%im2 = zeros(opts.imageSize(1), opts.imageSize(2), 3, numel(batch), 'single') ;
for i= 2:2:length(batch)
    I = single(imdb.images.data{1,batch(i)});
    
    imSize= size(I);
    centerx = randi(round([imSize(1)/4, 3*imSize(1)/4]),1);
    centery = randi(round([imSize(1)/4, 3*imSize(1)/4]),1);
    
    
    %wl = randi([round(imSize(1)/8), round(imSize(1)/2)]);
    %wr = randi([round(imSize(1)/8), round(imSize(1)/2)]);
    %hu = randi([round(imSize(1)/8), round(imSize(1)/2)]);
    %hd = randi([round(imSize(1)/8), round(imSize(1)/2)]);
    
    theta = deg2rad(randi(9,1) * 10);
    R = [ cos(theta), sin(theta);
        - sin(theta), cos(theta)];
    
    width = 45* randi(2,1);
    box_rest = [0,0; 0, width; width,width; width,0]-  width/2;
    bbox = R * box_rest';
    bbox(1,:) = bbox(1,:) + centerx;
    bbox(2,:) = bbox(2,:) + centery;
    bbox = round(bbox);
    bbox = max(1, bbox);
    bbox = min(size(I, 1), bbox);
    %    bbox(3) = min(size(I, 2), bbox(3));
    %    bbox(4) = min(size(I, 1), bbox(4));
    c = bbox(1,:);
    r = bbox(2,:);
    BW = roipoly(I,c,r);
    idx = find(BW==1);
    col = randi(3,1)* 128-128;
  
    r = I(:,:,1);
    g = I(:,:,2);
    b = I(:,:,3);
    
    r(idx) = col;
    g(idx) = col;
    b(idx) = col;
    
    I = cat(3,r,g,b);
    I = imresize(I, opts.imageSize);
    im(:,:,:,i) = bsxfun(@minus, I, offset)/256;
    
end

lab = imdb.images.labels(1,batch);
for i= 1:length(batch)
    R = Axis2Rot(lab{1,i});
    lab{1,i}= single(R(:));
    
end


if nargout > 0
    if useGpu
        im = gpuArray(im) ;
    end
    
    inputs = {'input', im,'label', lab};
    
end
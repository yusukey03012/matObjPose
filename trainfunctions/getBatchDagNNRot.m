function inputs = getBatchDagNNRot(imdb, batch, mode, opts, useGpu)

im = 128*ones(opts.imageSize(1), opts.imageSize(2), 3, numel(batch), 'single') ;
offset = opts.averageImage ;
for i= 1:length(batch)
    I = single(imdb.images.data{1,batch(i)});
    
    
    
    I = imresize(I, opts.imageSize);
    
    im(:,:,:,i) = bsxfun(@minus, I, offset) ;
    im(:,:,:,i) = im(:,:,:,i)./256;
    
    im_ = im(:,:,:,i);
    dx = randi(21) - 11 ;
    dy = randi(21) - 11 ;
    
    cx = 11:opts.imageSize(1)-11;
    cy= cx;
    sx = cx + dx;
    sy = cx + dy;
    im(sy,sx,:,i) = im_(cx,cy,:) ;
    
end

lab = imdb.images.labels(1,batch);
for i= 1:length(batch)
    lab{1,i}= single(lab{1,i});
end

lab2 = imdb.images.labels(2,batch);

lab3 = imdb.images.class_num(batch,1);


if nargout > 0
    if useGpu
        im = gpuArray(im) ;
    end
    
    inputs = {'input', im,'label', lab};
    
end
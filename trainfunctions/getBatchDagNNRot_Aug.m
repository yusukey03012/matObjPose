function inputs = getBatchDagNNRot_Aug(imdb, batch, mode, opts, useGpu)

im = 128* ones(opts.imageSize(1), opts.imageSize(2), 3, numel(batch), 'single') ;
offset = opts.averageImage ;
    for i= 1:length(batch)
        
        I = single(imdb.images.data{1,batch(i)});
        
        %{
        if strcmp(mode,'train')
        
        if rem(i,2) == 0
            
            a = 0.05;
            b = 0.1;
            r = (b).*rand(1,1) + a;
             I = imresize(I, opts.imageSize/2);
            I = imnoise(I,'salt & pepper',r);
            
        end
        
        end
        I = single(I);
        %}
        
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
    %{
    %im2 = zeros(opts.imageSize(1), opts.imageSize(2), 3, numel(batch), 'single') ;
    for i= 2:2:length(batch)
        I = single(imdb.images.data{1,batch(i)});
        
        imSize= size(I);
        
        center = round(imSize(1)/2);
        
        wl = randi([round(imSize(1)/8), round(imSize(1)/2)]);
        wr = randi([round(imSize(1)/8), round(imSize(1)/2)]);
        hu = randi([round(imSize(1)/8), round(imSize(1)/2)]);
        hd = randi([round(imSize(1)/8), round(imSize(1)/2)]);
        
        %x = randi(round(imSize(1)/2));
        %y = randi(round(imSize(1)/2));
        %w = randi([round(imSize(1)/3), (imSize(1)-x)]);
        %h = randi([round(imSize(1)/3), (imSize(1)-y)]);
        %bbox = [x, y, x+w, y+h];
        bbox = center + [-wl, -hu, wr, hd];

        bbox(1) = max(1, bbox(1));
        bbox(2) = max(1, bbox(2));
        bbox(3) = min(size(I, 2), bbox(3));
        bbox(4) = min(size(I, 1), bbox(4));
        
        
        I = I(bbox(2):bbox(4), bbox(1):bbox(3),:);
        
        diff = round((size(I,1)-size(I,2))/2);
        if diff>0 %pad width
            sq_img = padarray(I,[0,diff],128);
        else
            sq_img = padarray(I,[-diff,0],128);
        end
        [sq_imdim1, sq_imdim2, ~] = size(sq_img);
        
        I = imresize(sq_img, opts.imageSize);
             
        im(:,:,:,i) = bsxfun(@minus, I, offset)/256;
        %im2(:,:,:,i) = im2(:,:,:,i)./256;
        %dx = randi(21) - 11 ;
        %dy = randi(21) - 11 ;
        %cx = 11:opts.imageSize(1)-11;
        %cy= cx;
        %sx = cx + dx;
        %sy = cx + dy;
        %im(sy,sx,:,i) = im_(cx,cy,:);
    end
    
    %im(:,:,:,2:2:end) = im2(:,:,:,2:2:end);
    %}
    lab = imdb.images.labels(1,batch);
    for i= 1:length(batch)
        lab{1,i}= single(lab{1,i});
    end
    
    %lab = [lab,lab];
    
    lab2 = imdb.images.labels(2,batch);
    %lab3 = imdb.images.class_num(batch,1);
    
    %im = cat(4,im,im2);
    
    if nargout > 0
        if useGpu
            im = gpuArray(im) ;
        end
        
        inputs = {'input', im,'label', lab};
        
    end
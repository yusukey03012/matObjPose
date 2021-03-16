function Y = vl_nntukeyloss(X,c,iter,scbox, dzdy, varargin)

%Created by Vasileios Belagiannis.
%Contact: vb@robots.ox.ac.uk
%Only the l2loss for the heatmaps is included. The tukeyloss is in the
%following repo: https://github.com/bazilas/matconvnet-deepReg

opts.loss = 'l2loss' ;
opts.lossWeight=1;
opts.ignOcc=0;
opts = vl_argparse(opts,varargin) ;

switch lower(opts.loss)
    %l2loss
     case {'tukeyloss'}
        X = reshape(X(1,1,:,:),[size(c{1,1},1),size(c,2)]);
        Y = [c{1,1:size(c,2)}];
        
        %residuals
        res=(Y-X);
        
        %Median absolute deviation (MAD)
        MAD = 1.4826*mad(res',1)';
        
        %inliers (percentage of inliers)
        nonZer = round(100*sum(abs(res(:))<4.685)/numel(res));
        
        if iter<50 %(as in the paper)
        %if nonZer<70 %(similar to the above) - test it again
            MAD=MAD*7; %helps the convergence at the first iterations
        end
        
        res=bsxfun(@rdivide,res,MAD);
        c=4.685;
        
        if isempty(dzdy) %forward
            
            %tukey's beiweight function
            %(http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html)
            yt = (c.^2/6) * (1 - (1-(res./c).^2).^3);
            yt(find(abs(res)>c))=(c^2)/6;
            
            Y = sqrt(sum(yt(:)));
        else
            
            %derivatives
            tu= -1.*res.*((1-(res./c).^2).^2);
            
            Y_(1,1,:,:)= tu.*bsxfun(@lt,abs(res),c); % abs(x) < c
            
            Y = single (Y_ * dzdy);   
        end
    
    case {'l2loss'}
             
        X = reshape(X(1,1,:,:),[size(c{1,1},1),size(c,2)]);
        Y = [c{1,1:size(c,2)}];
        
        res=(Y-X);
        
        n=1;
        if isempty(dzdy) %forward
            Y = (sum(res(:).^2))/numel(res)*1000;
        else
            Y_(1,1,:,:)= -1.*(Y-X);
            Y = single (Y_ * (dzdy / n) );
        end
            
    case {'l2loss-heatmap'}
        if strcmp(opts.loss,'l2loss-heatmap')
            if iscell(c)
            Y = cat(4,c{2,:});
            else
            Y = c;
            c = opts.labels;
            end
            weight_mask = cat(4,c{3,:});
        end
        
        res=(Y-X);
        
        %missing annotation - zeros contribution 
        idx=repmat(sum(sum(Y,1),2)==0,size(res,1),size(res,2));
        res(idx)= zerosLike(res(idx)); %check it again!!!
        
        %n=sqrt(sum(res(:))); %L2 with square root
        n=1;
        if isempty(dzdy) %forward
            Y = sqrt(sum(res(:).^2))/(size(res,1)*size(res,2)*size(res,3)) *1000;%scale factor
        else
            %occluded keypoints - ignore them
            if opts.ignOcc
            idxOcc=Y<0;
            res(idxOcc)= zerosLike(res(idxOcc));
            end
                        
            res=weight_mask.*res;
            Y_= -1.*res;
            Y = single (Y_ * (dzdy / n) );
        end
        
    %error layer
    case {'mpe'} %mean pixel error
                X_orig = X;
        X = reshape(X(1,1,:,:),[size(c{1,1},1),size(c,2)]);
        Y = [c{1,1:size(c,2)}];
        
        if isempty(dzdy) %forward
            
            %residuals
            err=abs(Y-X);
            
            %scale back to pixels
            funScale = @(A,B) A.*(B);
            err = bsxfun(funScale,err,scbox);
            Y=[];
            Y = sum(err)./size(X,1);%error per samples
            Y = sum(Y);%summed batch error
            
        else %nothing to backprop
            Y = zerosLike(X_orig) ;
        end
    case {'mae-heatmap'} %mean absolute error
   
    case {'mse-heatmap'} %mean squarred error
        %GT stored in sparse matrices stacked next to each other
        if strcmp(opts.loss,'mse-heatmap')
            Y = cat(4,c{2,:});
        end
        
        if isempty(dzdy) %forward
            
            fun = @(A,B) A-B;
            err = bsxfun(fun,Y,X);
            
            %missing annotation - zeros contribution
            idx=repmat(sum(sum(Y,1),2)==0,size(err,1),size(err,2));
            err(idx)= zerosLike(err(idx)); %check it again!!!
            
            %occluded keypoints - ignore them
            if opts.ignOcc
            idxOcc=Y<0;
            err(idxOcc)= zerosLike(err(idxOcc));
            end
            
            Y = sum(err(:).^2)/(size(X,1)*size(X,2)*size(X,3));%error per batch / not per samples
            
        else %nothing to backprop
            Y = zerosLike(X) ;
        end
        
    otherwise
        error('Unknown loss ''%s''.', opts.loss) ;
end

% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
    y = gpuArray.zeros(size(x),'single') ;
else
    y = zeros(size(x),'single') ;
end


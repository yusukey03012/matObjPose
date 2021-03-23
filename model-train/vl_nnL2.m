<<<<<<< HEAD
function Y = vl_nnL2(X,c,dzdy)

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

=======
function Y = vl_nnL2(X,c,dzdy)

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

>>>>>>> 32b67e2fe1fbf86b60b67a08d8c4f76c21baf043

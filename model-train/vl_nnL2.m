
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



function A = Axis2Rot(b)		
%#codegen
iter = size(b,2);
A =zeros(3*iter,3); 
for i = 1:iter

 
theta = sum(b(:,i).^2).^0.5;
A(3*i-2:3*i,:) = eye(3);
w = [0,-b(3,i),b(2,i);
     b(3,i),0,-b(1,i);
     -b(2,i),b(1,i),0];
 
if (theta ~= 0)
    
    A(3*i-2:3*i,:) = A(3*i-2:3*i,:)...
        + (sin(theta) / theta) * w...
        + ((1 - cos(theta)) / theta^2) * w^2;
end

end
function [R_out, s_out, T_out] = PolarDecomposition(T_in,mode)
%#codegen

R_out=zeros(size(T_in));
s_out = zeros(size(T_in,1)/3,1);
T_out =[];

for i=1:size(T_in,1)/3                            
            
            [u,k,v]=svd(T_in(3*i-2:3*i,:));
            
            D = diag([1,1,det(u*v')]);
   
            R=u*v';
            s = ((k(1,1)^2+k(2,2)^2+k(3,3)^2)/3).^0.5;
            
            R_out(3*i-2:3*i,:) = R;
            s_out(i) = s;
                                            
end

if strcmp(mode,'similarity') ==1
T_out=zeros(size(T_in));
for i=1:size(T_in,1)/3                            
            
            T_out(3*i-2:3*i,:) = s_out(i)*R_out(3*i-2:3*i,:);
                    
end
end


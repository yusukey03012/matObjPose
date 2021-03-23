function R = Rotation_by_Axis(ax,theta)

theta = theta/180*pi;
if strcmp(ax, 'x')
    
    R = [1, 0, 0;
         0, cos(theta), -sin(theta);
         0, sin(theta), cos(theta)];
elseif strcmp(ax, 'y')
    
    R = [  cos(theta), 0, sin(theta);
           0, 1, 0;
         - sin(theta), 0, cos(theta)];

elseif strcmp(ax, 'z')
    
        R = [cos(theta), -sin(theta), 0;
             sin(theta),  cos(theta), 0;
             0, 0, 1];

end
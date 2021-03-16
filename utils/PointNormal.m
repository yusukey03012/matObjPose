classdef PointNormal < matlab.mixin.Copyable
    %UNTITLED8 ���̃N���X�̊T�v�������ɋL�q
    %   �ڍא����������ɋL�q
    
    properties
        
        vertex = []       
        edgeIdx = []
        numVertex =[]
        
        vNormal =[]
                bbox_min = []
        bbox_max = []
        bbox_r = []
    end
    
    methods
        
        
        function self = PointNormal(vertex,edgeIdx,Normal)
            
            self.vertex = vertex;    
            self.numVertex = length(self.vertex);
            self.edgeIdx = edgeIdx;
            self.vNormal = Normal;
            
            self.bbox_min=min(self.vertex);
            self.bbox_max=max(self.vertex);
            self.bbox_r=norm(self.bbox_max-self.bbox_min);
          
        end
               
        
    end
    
end


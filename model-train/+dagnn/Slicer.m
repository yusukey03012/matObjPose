classdef Slicer < dagnn.ElementWise
    properties
        
        height = []
        width = []
       
    end
    
    methods
        function outputs = forward(self, inputs, params)
            
            [h,w,ch,bs] = size(inputs{1});
            ch2= ch/self.width/self.height;           
            outputs{1} = reshape(inputs{1}, [self.height, self.width, ch2, bs]);
           
            
        end
        
        function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
                        
           
            [h,w,ch,bs] = size(inputs{1});      
            ch2 = ch * self.width * self.height;  
            derInputs{1} = reshape(derOutputs{1}, [1,1, ch,bs]);
            derParams = {} ;
            
        end
        
        function obj = Slicer (varargin)
            obj.load(varargin) ;
        end
    end
end

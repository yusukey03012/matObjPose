classdef Flatten5 < dagnn.ElementWise
    properties
        
    end
    
    methods
        function outputs = forward(self, inputs, params)
            
            [h,w, ch, sz] = size(inputs{1});
                       
            outputs{1} = reshape(inputs{1}, [1,1,h*w*ch,sz]);
            
        end
        
        function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
            
            derInputs{1} = reshape(derOutputs{1}, size(inputs{1}));
            derParams = {} ;
            
        end
        
        function obj = Flatten5(varargin)
            obj.load(varargin) ;
        end
    end
end

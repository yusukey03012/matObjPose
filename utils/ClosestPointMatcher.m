classdef ClosestPointMatcher < matlab.mixin.Copyable
    %CLOSESTPOINTMATCHER このクラスの概要をここに記述
    %   詳細説明をここに記述
    
    properties
        
        target = []
        source = []
        KDT = []
        
    end
    
    
    
    methods(Access='public')
        
        
        function self = ClosestPointMatcher(source,target)
            
            self.source = source;
            self.target = target;
            self.KDT = KDTreeSearcher(target.vertex);              
           
            
        end
        function delete(self)         
            
        end
        
        function [idxS, idxT, goalPos] = ComputeSourceToTargetMatch(self, option)
            
            if option.visibility==1
            
                vec = bsxfun(@minus, option.viewpoint,self.source.vertex);
                l=sum(vec.^2,2).^0.5;
                dir=vec./[l,l,l];
                d= dot(dir,self.source.vNormal,2);
                
                idxVisible =find(d>0);
                
            else
                
                idxVisible = 1:length(self.source.vertex);
            end
            
            
            
            idx = knnsearch(self.KDT, self.source.vertex(idxVisible,:) ,'K',1);
            disp = self.target.vertex(idx,:) - self.source.vertex(idxVisible,:);
            dst = sum(disp.^2,2).^0.5;
           
            % rejection
            theta = acos(sum(self.target.vNormal(idx,:) .* self.source.vNormal(idxVisible,:),2))./pi * 180;
            idxCorrect = find(( dst' < option.dst * self.target.bbox_r ) & (theta < option.theta )' );
          
              
            idxS = idxVisible(idxCorrect);
            idxT = idx(idxCorrect);
            
            % normal projection
            if option.project ==1
            
                d = sum(self.source.vNormal(idxS,:).*disp(idxCorrect,:),2);
                disp_project = self.source.vNormal(idxS,:).*[d,d,d];          
                goalPos = self.source.vertex(idxS,:) + disp_project;
            
            else
            
                goalPos = self.target.vertex(idxT,:);
                
            end           
            
            
        end
        
        
        function [idxS, idxT, goalPos] = ComputeTargetToSourceMatch(self, option)
                

            if option.visibility==1
            
                vec = bsxfun(@minus, option.viewpoint,self.source.vertex);
                l=sum(vec.^2,2).^0.5;
                dir=vec./[l,l,l];
                d= dot(dir,self.source.vNormal,2);
                
                idxVisible =find(d>0);
                
            else
                
                idxVisible = (1:length(self.source.vertex))';
            end
            
            
           
            idx = knnsearch(self.source.vertex(idxVisible,:), self.target.vertex ,'K',1);           
                                 
                        
            disp = self.target.vertex - self.source.vertex(idxVisible(idx),:);
            dst = sum(disp.^2,2).^0.5;
           
            % rejection
            theta = acos(sum(self.target.vNormal .* self.source.vNormal(idxVisible(idx),:),2))./pi * 180;
            idxCorrect = find(( dst' < option.dst * self.target.bbox_r) & (theta < option.theta )' );
            
            [s, sid]= sortrows([idxVisible(idx(idxCorrect)),dst(idxCorrect) ]);
            
            
            %s = flipud(s);
            %sid = flipud(sid);
            
            
            [u,uid] = unique(s(:,1));
            
            idxS = u;
            idxT = idxCorrect(sid(uid));
            %idxS = idx(idxCorrect);
            %idxT = idxCorrect;
                        
            % normal projection
            if option.project ==1
            
                d = sum(self.source.vNormal(idxS,:).*disp(idxT,:),2);
                disp_project = self.source.vNormal(idxS,:).*[d,d,d];          
                goalPos = self.source.vertex(idxS,:) + disp_project;
            
            else
            
                goalPos = self.target.vertex(idxT,:);
                
            end    
            
            
            
        end
        
        
        
    end
    
end


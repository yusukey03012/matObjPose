classdef Mesh < matlab.mixin.Copyable
    % Mesh このクラスの概要をここに記述
    %   詳細説明をここに記述
    
    properties
        
        vertex = []
        face = []
        
        mode =[]
        
        bbox_min = []
        bbox_max = []
        bbox_r = []
        
        numVertex = []
        numEdge = []
        numFace = []
        numPair = []
        numBoundary =[]
        
        edgeIdx =[]
        fRing=[]
        vRing=[]
        vEdge =[]
        triPair = []
        hingePair  = []
        validPair = []
        
        
        edgeVec = []
        edgeLength = []
        edgeDir = []
        
        
        edgeVecV = []
        edgeLengthV = []
        
        edgeVec1 = []
        edgeVec2 = []
        
        centroids = []
        area = []
        theta = []
        hinge  = []
        
        l1 = []
        l2 = []
        fNormal = []
        vNormal = []
        
        
        idxBoundary =[]
        pairBoundary = []
        
        L = []
        A = []
        
    end
    
    methods (Access='public')
        
        function self=Mesh(vertex, face, mode)
            
            self.vertex = vertex;
            self.face = face;
            self.mode = mode;
            self.Setup()
            
            
        end
        
        function Setup(self)
            
            self.bbox_min=min(self.vertex);
            self.bbox_max=max(self.vertex);
            self.bbox_r=norm(self.bbox_max-self.bbox_min);
            self.ComputeCentroids()
            
            self.numVertex = size(self.vertex,1);
            self.numFace = size(self.face,1);
            self.ConstructEdgeConnectivity()
            self.numEdge = size(self.edgeIdx,1);
            if strcmp(self.mode,'simple')~=1
                
                
                %self.ConstructTriPair()
                self.ComputeVertexRing()
                self.ComputeEdgeVectorV()
                self.ComputeEdgeLengthV()
                
                self.ComputeEdgeVector()
                self.ComputeEdgeLength()
                
                
                self.numPair = size(self.triPair,1);
                
                %self.ComputeAngle()
                %self.ComputeArea()
                %       self.ComputeDihedralAngle()
                %     self.ConstructHingeVertexPair()
                % self.ComputeVertexFaceRing()
                %self.ComputeBoundary()
                %self.ComputeVertexRing()
                
            end
            
        end
        
        
        
        function Update(self,newVertex)
            
            
            self.vertex = newVertex;
            
            self.bbox_min=min(self.vertex);
            self.bbox_max=max(self.vertex);
            self.bbox_r=norm(self.bbox_max-self.bbox_min);
            
            self.ComputeCentroids()
            %self.ComputeNormal()
            
            if strcmp(self.mode,'simple')~=1
                
                self.ComputeEdgeVectorV()
                self.ComputeEdgeLengthV()
                
                self.ComputeEdgeVector()
                self.ComputeEdgeLength()
                self.ComputeAngle()
                self.ComputeArea()
                %{
                self.ComputeDihedralAngle()
                %}
            end
            
        end
        
        
        function ComputeCentroids(self)
            
            self.centroids = (self.vertex(self.face(:,1),:) +...
                self.vertex(self.face(:,2),:) +...
                self.vertex(self.face(:,3),:))/3;
            
        end
        
        
        function edgeTransformed = BackTransformEdge(self,template)
            R = MeshFunction.ComputeOptimalRotation_mex(template.vertex, self.vertex, template.vRing);
            edgeTransformed  = self.edgeVecV;
            for i = 1:length(self.edgeVecV)
                edgeTransformed(i,:) = edgeTransformed(i,:) * R(3*self.vEdge(i,1)-2:3*self.vEdge(i,1),:)';
            end
        end
        
        function ComputeEdgeDir(self,template)
            edgeTransformed = self.BackTransformEdge(template);
            self.edgeDir = edgeTransformed ./[self.edgeLengthV, self.edgeLengthV, self.edgeLengthV];
        end
        
        function ComputeLaplacianMatrix(self)
            
            [self.L, a] = mshlp_matrix2(self.vertex,self.face);
            self.A = spdiags(a,0,size(a,1),size(a,1));
            
        end
        
        
        
        function ConstructHingeVertexPair(self)
            
            self.hingePair = zeros(self.numPair,2);
            for i = 1:self.numPair
                a = self.face(self.triPair(i,1),:);
                b = self.face(self.triPair(i,2),:);
                
                self.hingePair(i,1) = setdiff(a,self.edgeIdx(self.validPair(i),:));
                self.hingePair(i,2) = setdiff(b,self.edgeIdx(self.validPair(i),:));
            end
        end
        function ComputeVertexRing(self)
            
            nverts = max(max(self.face));
            
            F=[self.face(:,1:2);self.face(:,2:3);self.face(:,[3,1])];
            F = [F;fliplr(F)];
            F=sortrows(F);
            %A = MeshClass.triangulation2adjacency(self.face');
            %[i,j,s] = find(sparse(A));
            
            
            
            % create empty cell array
            %self.vRing = cell(nverts,1);
            self.vRing = struct('r', num2cell(zeros(1,nverts)));
            
            
            self.vEdge = zeros(10 * nverts,2);
            
            start=1;
            for m = 1:nverts
                
                onering=unique(F(F(:,1)==m,2));
                %self.vRing{m} = onering;
                
                self.vRing(m).r = onering;
                
                self.vEdge(start:start + length(onering)-1,1) = m;
                self.vEdge(start:start + length(onering)-1,2) = onering;
                start = start+ length(onering);
            end
            self.vEdge = self.vEdge(1:start-1,:);
            
            
        end
        
        function ComputeVertexRing2(self,idx)
            
            nverts = length(idx);
            
            F=[self.face(:,1:2);self.face(:,2:3);self.face(:,[3,1])];
            F = [F;fliplr(F)];
            F=sortrows(F);
            %A = MeshClass.triangulation2adjacency(self.face');
            %[i,j,s] = find(sparse(A));
            
            % create empty cell array
            self.vRing = struct('r', num2cell(zeros(1,nverts)));
            
            for m = 1:nverts
                
                onering=unique(F(F(:,1)== idx(m),2));
                
                self.vRing(m).r = onering;
                
            end
            
            
        end
        
        function ComputeVertexFaceRing(self)
            
            
            self.fRing=cell(self.numVertex,1);
            
            for i= 1:self.numVertex
                
                [r,c] = find(self.face == i);
                self.fRing{i} = r;
                
            end
            
            
        end
        
        function ComputeVertexFaceRing2(self,idx)
            
            
            self.fRing=cell(length(idx),1);
            
            for i= 1:length(idx)
                
                I = idx(i);
                [r,c] = find(self.face == I);
                self.fRing{i} = r;
                
            end
            
            
        end
        
        
        function ComputeEdgeVector(self)
            
            self.edgeVec = self.vertex(self.edgeIdx(:,2),:)...
                - self.vertex(self.edgeIdx(:,1),:);
            
        end
        
        function ComputeEdgeLength(self)
            
            self.edgeLength = sum(self.edgeVec.^2,2).^0.5;
            
        end
        
        function ComputeEdgeVectorV(self)
            
            self.edgeVecV = self.vertex(self.vEdge(:,2),:)...
                - self.vertex(self.vEdge(:,1),:);
            
        end
        
        function ComputeEdgeLengthV(self)
            
            self.edgeLengthV = sum(self.edgeVecV.^2,2).^0.5;
            
        end
        
        function ComputeAngle(self)
            
            idx = [3:3:self.numEdge;1:3:self.numEdge;2:3:self.numEdge];
            idx = idx(:);
            
            edgeVec2 = - self.edgeVec(idx,:);
            edgeLength2 = self.edgeLength(idx,:);
            edgeVec_n = self.edgeVec./[self.edgeLength,self.edgeLength,self.edgeLength];
            edgeVec2_n = edgeVec2./[edgeLength2,edgeLength2,edgeLength2];
            
            d=sum(edgeVec_n .*edgeVec2_n ,2);
            self.theta = acos(d);
            
            
        end
        
        %------------------------------
        function ComputeNormal(self,varargin)
            
            self.edgeVec1=self.vertex(self.face(:,2),:)-self.vertex(self.face(:,1),:);
            self.edgeVec2=self.vertex(self.face(:,3),:)-self.vertex(self.face(:,1),:);
            
            self.l1=sqrt(sum(self.edgeVec1.^2,2));
            self.l2=sqrt(sum(self.edgeVec2.^2,2));
            
            N = cross(self.edgeVec1,self.edgeVec2);
            d = sum(N.^2,2).^0.5;
            self.fNormal=N./[d,d,d];
            
            normal=zeros(length(self.vertex),3);
            for i=1:length(self.face)
                for j=1:3
                    normal(self.face(i,j),: ) = normal( self.face(i,j),: ) + N(i,:);
                end
            end
            
            if isempty(varargin) ==1
                
                
                
                
                %normal= ComputeNormal_mex(self.vertex,self.face,N);
                
                d2=sum(normal.^2,2).^0.5;
                
                self.vNormal= normal./[d2,d2,d2];
            end
            
        end
        
        function ComputeArea(self)
            
            self.area = self.edgeLength(3:3:end).*self.edgeLength(1:3:end).*sin(self.theta(1:3:end))/2;
            
        end
        
        function ComputeDihedralAngle(self)
            
            w = cross(self.fNormal(self.triPair(:,1),:),self.fNormal(self.triPair(:,2),:),2);
            
            s = sum(w.* self.edgeVec(self.validPair,:),2);
            
            sig = sign(s);
            d = dot(self.fNormal(self.triPair(:,1),:),self.fNormal(self.triPair(:,2),:),2);
            
            c = d .* self.edgeLength(self.validPair);
            
            hinge1 = sig.*acos(d);
            
            hinge2 = atan2(s,c);
            
            self.hinge = hinge2;
            
        end
        
        function ConstructTriPair(self)
            
            trep = TriRep(self.face, self.vertex);
            pair = edgeAttachments(trep, self.edgeIdx);
            pairMat = zeros(length(pair),2);
            numPair = 0;
            validPair =  zeros(length(pair),1);
            for i= 1:length(pair)
                
                if length(pair{i})==2
                    
                    numPair= numPair+1;
                    pairMat(numPair,:) = pair{i};
                    validPair(numPair) = i;
                    
                end
                
            end
            
            self.triPair = pairMat(1:numPair,:);
            self.validPair = validPair(1:numPair);
            
        end
        
        
    end
    
    
    
    methods (Access='private')
        
        
        function ComputeBoundary(self)
            
            
            t = TriRep(self.face, self.vertex);
            [edge x] = freeBoundary(t);
            
            self.idxBoundary = flann64.flann_search(single(self.vertex'),single(x'),1,...
                struct('checks',128,'algorithm','kmeans','branching',64,'iterations',1));
            
            self.numBoundary = length(self.idxBoundary);
            
            self.pairBoundary = self.idxBoundary(edge);
            self.pairBoundary = [self.pairBoundary;fliplr(self.pairBoundary)];
            self.pairBoundary = sortrows(self.pairBoundary);
            
            
        end
        
        
        function ConstructEdgeConnectivity(self)
            
            self.edgeIdx = [self.face(:,1:2),self.face(:,2:3),self.face(:,[3,1])]';
            self.edgeIdx =reshape(self.edgeIdx,2,[])';
        end
        
        
        
        
        
        
    end
    
    
    
end


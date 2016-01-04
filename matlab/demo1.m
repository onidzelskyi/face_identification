function [] = demo1(group, single)
% Face identification using by euclidian distance
% Identify person from group of persons
% group - photo of a group of persons
% single - photo of a single person which face identified from group

% default options
% group - group image file
if ~exist('group','var'), group = '../test/test1g.jpeg'; end

% single - single image file
if ~exist('single','var'), single = '../test/test1s.jpeg'; end

% face' size
M = 0;
N = 0;

% Black box
faceDetector = vision.CascadeObjectDetector;

%% 1. Group
% 1.1. Read image
% 1.2. Extract faces from image
% 1.3. Unfold faces matrices to a vectors
g = [];
G = imread(group);
gboxes = step(faceDetector, G);
if size(gboxes,1)
% Set face' size
M = floor(mean(gboxes(:,3)));
N = M;
% extract faces
for i=1:size(gboxes,1)
    p = imresize(G(gboxes(i,2):gboxes(i,2)+gboxes(i,4),gboxes(i,1):gboxes(i,1)+gboxes(i,3)),[M,NaN]);
    g = [g; double(reshape(p,[1, size(p,1)^2]))];
end

end
%% 2. Single
% 2.1. Read image
% 2.2. Extract faces from image
% 2.3. Unfold faces matrices to a vectors
% 2.4. Normalize faces
S = imread(single);
sboxes = step(faceDetector, S);
s = [];
if size(sboxes,1)
    s = imresize(S(sboxes(1,2):sboxes(1,2)+sboxes(1,4),sboxes(1,1):sboxes(1,1)+sboxes(1,3)),[M,NaN]);
    s = double(reshape(s,[1,size(s,1)^2]));
end

if size(g,1) && size(s,1)
    %% 4. Identification
    d = bsxfun(@minus, g,s);
    dist = arrayfun(@(idx) norm(d(idx,:)), 1:size(d,1)).^2
    [a,b] = min(dist);
    GFaces = insertObjectAnnotation(G, 'rectangle', gboxes(b,:), 'Target');
    SFaces = insertObjectAnnotation(S, 'rectangle', sboxes(1,:), 'Source');
    imwrite(GFaces, 'G.jpeg'); imwrite(SFaces, 'S.jpeg');
    %subplot(1,2,1); subimage(GFaces);
    %subplot(1,2,2); subimage(SFaces);
else
    fprintf('Error\n');
end
end
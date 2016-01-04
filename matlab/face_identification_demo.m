function [] = face_identification_demo(group, single)
%% Identify person from group of persons
% group - photo of a group of persons
% single - photo of a single person which face identified from group

% Clear workspace
%clear all; clc; close all;

% default options
% group - group image file
if ~exist('group','var'), group = '../test/g3.jpg'; end

% single - single image file
if ~exist('single','var'), single = '../test/s3.jpg'; end

% face' size
M = 0;
N = 0;

apply_lbp = 0;
lbp_cells = 24;
lbp_radius = 2;

% Black box
faceDetector = vision.CascadeObjectDetector;

%% 1. Group
% 1.1. Read image
% 1.2. Extract faces from image
% 1.3. Unfold faces matrices to a vectors
% 1.4. Normalize faces
g = [];
G = imread(group);
% convert to gray
G_gray = rgb2gray(G);

gboxes = step(faceDetector, G);
% Set face' size
M = floor(mean(gboxes(:,3))); N = M;

% extract faces
for i=1:size(gboxes,1)
    p = imresize(G_gray(gboxes(i,2):gboxes(i,2)+gboxes(i,4),gboxes(i,1):gboxes(i,1)+gboxes(i,3)),[M,NaN]);
    g = [g; double(reshape(p,[1, size(p,1)^2]))];
end
% Normalize images
%g = bsxfun(@minus, g, mean(g));

%% 2. Single
% 2.1. Read image
% 2.2. Extract faces from image
% 2.3. Unfold faces matrices to a vectors
% 2.4. Normalize faces
S = imread(single);
S_gray = rgb2gray(S);
sboxes = step(faceDetector, S);
s = [];
for i=1:size(sboxes,1)
    p = imresize(S_gray(sboxes(i,2):sboxes(i,2)+sboxes(i,4),sboxes(i,1):sboxes(i,1)+sboxes(i,3)),[M,NaN]);
    s = [s; double(reshape(p,[1,size(p,1)^2]))];
    % Normalize images
    %s = bsxfun(@minus, s, mean(s));
end

%% LBP
group_lbp = [];
single_lbp = [];
if apply_lbp
    for i=1:size(g,1)
        I = reshape(g(i,:), M, N);
        feature = extractLBPFeatures(I, 'CellSize', [floor(M/lbp_cells), floor(N/lbp_cells)], 'Radius', lbp_radius);
        group_lbp = [group_lbp; feature];
    end
    for i=1:size(s,1)
        I = reshape(s(i,:), M, N);
        feature = extractLBPFeatures(I, 'CellSize', [floor(M/lbp_cells), floor(N/lbp_cells)], 'Radius', lbp_radius);
        single_lbp= [single_lbp; feature];
    end
    g = group_lbp;
    s = single_lbp;
end


%% 4. Identification
for i=1:size(s, 1)
    d = bsxfun(@minus, g,s(i,:));
    dist = arrayfun(@(idx) norm(d(idx,:)), 1:size(d,1)).^2;
    %dist = calcDistance(g, s(i,:),3);
    [a,b] = min(dist)
    GFaces = insertObjectAnnotation(G, 'rectangle', gboxes(b,:), 'Target');
    SFaces = insertObjectAnnotation(S, 'rectangle', sboxes(1,:), 'Source');
    imwrite(GFaces, 'G.jpeg'); imwrite(SFaces, 'S.jpeg');
    %subplot(1,2,1); subimage(GFaces);
    %subplot(1,2,2); subimage(SFaces);
end

end
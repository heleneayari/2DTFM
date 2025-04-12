function [ sp, conver, erreur] = pyrLK_2D( pyr1, pyr2, pts0, winSize, maxIter, threshold)
% [ speed failure error ] = pyrLK( img1, img2, features, windowSize=5, ...
%                                   maxIteration=2, stopThreshold=0.5 )
%
% pyrLK tracks 'features' from images in pyr1 to pyr2
% it is a matlab implementation of the iterative-pyramidal-Lucas-Kanade
% motion tracker described in [BOUGUET02]:
% "Pyramidal Implementation of the Lucas Kanade Feature Tracker:
%  Description of the algorithm"
%
% Input:
%  - pyr1 and pyr2 are pyramids of images made with 'makePyramide.m'
%  - feature is a Nx2 matrix containing the x,y coordinates of the features to track
%
%  - windowSize is the size of the square used to estimate local gradient
%  - iterate at most maxIteraction times per pyramid level
%  - or stop if convergence is less than stopThreshold (in pixels)
% * 'windowSize', 'maxIteration' and 'stopThreshold' can be scalars or
%   vectors with a different value for each pyramid level.
%
% Output:
%  - speed:   estimated speed of features (in lower pyramid image)
%  - failure: tracking failure (see below)
%  - error:   final difference of pixel color
%
% 'failure' is an array containing 2 counts of failures for each particle.
% The first is the number of times LK has failed due to a weak gradient of
% color intensity in the area around the features.
% The second is the number of times it has been tracked out of the image and
% force back inside, or lost due to algorithmic failure (this should not
% happened anymore, otherwise warnings are also displayed).
%
% See also: crKLT, crKLT_config, private/makePyramid


if nargin<3, error('pyrLK:not_enough_input','Not enough input'); end
if nargin<4, winSize   = 5;    end
if nargin<5, maxIter   = 2;    end
if nargin<6, threshold = 0.5;  end


pyrNumber = length(pyr1);

if length(winSize)<pyrNumber
    winSize = [ones(1,pyrNumber-length(winSize))*winSize(1) winSize];
end
if length(threshold)<pyrNumber
    threshold = [ones(1,pyrNumber-length(threshold))*threshold(1) threshold];
end
if length(maxIter)  <pyrNumber
    maxIter = [ones(1,pyrNumber-length(maxIter))*maxIter(1) maxIter];
end


% initializes some variables:
% sp: displacement (speed) of features
% ds: radius of neighborhood
if size(pts0,2)>=4
    sp   = pts0(:,3:4) ./ (2^pyrNumber); % init in upper pyramid level
    pts0 = pts0(:,1:2);
else
    sp = zeros(size(pts0,1),2);
end

ds = max(floor(winSize/2),1);
winSize = 2*ds +1;

erreur = zeros(size(pts0,1),pyrNumber);
for k=pyrNumber:-1:1
        
    
    % relative indices of the neighborhood of a feature
    clear r
    % initialize variables used in current pyramid level
    sp   = 2*sp;
    pts  = pts0./(2^(k-1));
    pts(:,:)=pts(:,:)+ones(length(pts),1)*[winSize(k) winSize(k)];
    img1 = pyr1(k).img;
    img2 = pyr2(k).img;
    Px   = pyr1(k).gradX;
    Py   = pyr1(k).gradY;
    [r(:,:,1),r(:,:,2)] = meshgrid(-ds(k):ds(k));
    r = [ reshape(r(:,:,1),winSize(k).^2,1) reshape(r(:,:,2),winSize(k).^2,1) ];
    
    % for each feature to track
    for i=1:size(pts,1)
        % initialize variable use for current feature
        count = 0;              % number of iteration
        ldspl = threshold(k)+1; % norm of last iteration displacement 'dsp'
        % function bilinear is implemented below
        a = [ bilinear(Px, [ pts(i,1)+r(:,1) pts(i,2)+r(:,2)]) ...
            bilinear(Py, [ pts(i,1)+r(:,1) pts(i,2)+r(:,2)]) ];
        
        
        A = a'*a;
        I = bilinear(img1(), [ pts(i,1)+r(:,1) pts(i,2)+r(:,2) ]);
        
        
        % iterate until it converges (or fails to)
        while count<maxIter(k) && ldspl>threshold(k)
            
            b = I - bilinear(img2(), [ pts(i,1)+sp(i,1)+r(:,1) pts(i,2)+sp(i,2)+r(:,2) ]);
            
            B = a'*b;
            
            dsp    = A\B;
            ldspl   = norm(dsp);
            sp(i,:) = sp(i,:) + dsp(2:-1:1)';
            
            
            count = count+1;
        end
        
        if ldspl<threshold(k)
            conver(i,k)=count;
        else
            conver(i,k)=0;
        end
        
    end
    
    
    
    
    
    
    % compute error if asked
    if nargout==3
        
        
        for i=1:size(pts0,1)
            I = bilinear(img1(), [ pts(i,1)+r(:,1) pts(i,2)+r(:,2) ]);
            b = I - bilinear(img2(), [ pts(i,1)+sp(i,1)+r(:,1) pts(i,2)+sp(i,2)+r(:,2) ]);
            erreur(i,k) = sum(b.^2,1)./sum(I.^2,1)*100;
        end
    end
end

% return values of pixels with non-integer indices 'ind' in 'array' using
% bilinear interpolation. Same purpose as 'interp2' but specific to our
% case and much quicker.
    function interp = bilinear( array, ind )
        ind_0  = floor(ind);                          % top    - left
        ind_x  = ind_0;  ind_x(:,1) = ind_x(:,1) +1;  % top    - right
        ind_y  = ind_0;  ind_y(:,2) = ind_y(:,2) +1;  % bottom - left
        ind_xy = ind_0;  ind_xy     = ind_xy     +1;  % bottom - right
        
        alpha  = ind - ind_0;
        beta   = 1 - alpha;
        
        % check if still in array
        [sx,sy] = size(array);
        
        % clamp indices
        ind_0 (ind_0 <1) = 1; ind_0 (ind_0 (:,1)>sx,1)=sx; ind_0 (ind_0 (:,2)>sy,2)=sy;
        ind_x (ind_x <1) = 1; ind_x (ind_x (:,1)>sx,1)=sx; ind_x (ind_x (:,2)>sy,2)=sy;
        ind_y (ind_y <1) = 1; ind_y (ind_y (:,1)>sx,1)=sx; ind_y (ind_y (:,2)>sy,2)=sy;
        ind_xy(ind_xy<1) = 1; ind_xy(ind_xy(:,1)>sx,1)=sx; ind_xy(ind_xy(:,2)>sy,2)=sy;
        
        % convert to 1D indices (same as sub2ind but quicker)
        ind_0 (:,1) = (ind_0 (:,2)-1)*sx + ind_0 (:,1);
        ind_x (:,1) = (ind_x (:,2)-1)*sx + ind_x (:,1);
        ind_y (:,1) = (ind_y (:,2)-1)*sx + ind_y (:,1);
        ind_xy(:,1) = (ind_xy(:,2)-1)*sx + ind_xy(:,1);
        
        % compute the sum of the bilinear interpolation
        % interp = zeros(size(alpha,1),1);
        
        interp =          beta (:,1).*beta (:,2).*array(ind_0 (:,1));
        interp = interp + alpha(:,1).*beta (:,2).*array(ind_x (:,1));
        interp = interp + beta (:,1).*alpha(:,2).*array(ind_y (:,1));
        interp = interp + alpha(:,1).*alpha(:,2).*array(ind_xy(:,1));
        
    end
end



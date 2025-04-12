function [ sp,conver,erreur] = pyrLK_2Dgpu( pyr1, pyr2, pts0gpu, winSize, maxIter, threshold)
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
% if size(pts0,2)>=4
%     sp   = pts0(:,3:4) ./ (2^pyrNumber); % init in upper pyramid level
%     pts0 = pts0(:,1:2);
% else
spgpu = gpuArray(single(zeros(size(pts0gpu))));

% end

ds = max(floor(winSize/2),1);
winSize = 2*ds +1;

conver=zeros(length(pts0gpu),pyrNumber);

for k=pyrNumber:-1:1
    %k
    % initialize variables used in current pyramid level
    spgpu   = 2*spgpu;
    ind0=false(length(pts0gpu),1);
    
    ptsgpu  = pts0gpu./(2^(k-1));
    ptsgpu=ptsgpu+gpuArray(single(ones(length(pts0gpu),1)*[winSize(k) winSize(k)]));
    
    
    img1 = gpuArray(single(pyr1(k).img));
    img2 = gpuArray(single(pyr2(k).img));
    Px   = gpuArray(single(pyr1(k).gradX));
    Py   = gpuArray(single(pyr1(k).gradY));
    [sx,sy] = size(img2) ;
    % relative indices of the neighborhood of a feature
    clear r rgpu matvois matvoisx matvoisy matvois2 matvoisx2 matvoisy2 b I a A B
    [r(:,:,1),r(:,:,2)] = meshgrid(-ds(k):ds(k));
    rgpu = gpuArray(single([ reshape(r(:,:,1),winSize(k).^2,1) reshape(r(:,:,2),winSize(k).^2,1) ]));

    ptsgpu=permute(ptsgpu,[3 2 1]);
    spgpu2=permute(spgpu,[3 2 1]);
    matvois=repmat(ptsgpu,length(rgpu),1,1)+repmat(rgpu,1,1,length(ptsgpu));
    matvoisx=matvois(:,1,:);
    matvoisy=matvois(:,2,:);
    matvois2=repmat(ptsgpu+spgpu2,length(rgpu),1,1)+repmat(rgpu,1,1,length(ptsgpu));
    matvoisx2=matvois2(:,1,:);
    matvoisy2=matvois2(:,2,:);
    
    matvoisx(matvoisx>sx)=sx;
    matvoisy(matvoisy>sy)=sy;
    
    matvoisx(matvoisx<1)=1;
    matvoisy(matvoisy<1)=1;
    
    
    % initialize variable use for current feature
    count = 0;              % number of iteration
    
    
    
    
    % function bilinear is implemented below
    
    a =  [arrayfun(@bilinearPx,matvoisx,matvoisy)...
        arrayfun(@bilinearPy,matvoisx,matvoisy)];

    
    A = pagefun(@mtimes,permute(a,[2 1 3]),a);

    
    I = arrayfun(@bilinearimg1,matvoisx,matvoisy);

    
    
    %         % iterate until it converges (or fails to)
    %     while count<maxIter(k) && ldspl>threshold(k)
    indf=ind0;
    ind1 = (conver(:,k)==0);
    while count<maxIter(k)
        matvoisx2(matvoisx2>sx)=sx;
        matvoisy2(matvoisy2>sy)=sy;
        matvoisx2(matvoisx2<1)=1;
        matvoisy2(matvoisy2<1)=1;
        b = I - arrayfun(@bilinearimg2,matvoisx2,matvoisy2);
        
        
        B=pagefun(@mtimes,permute(a,[2 1 3]),b);
        
        
        dsp    = pagefun(@mldivide,A,B);
        
        dsp2(1:size(dsp,3),[2 1])=squeeze(permute(dsp,[3 1 2]));
        
        ldspl   =  sqrt(dsp2(:,1).^2+dsp2(:,2).^2);
        ind=ldspl<threshold(k);
        
        indf=indf+ind;
        conver(indf==1,k)=count+1;
        
        
        
        spgpu(ind1,:) = spgpu(ind1,:)+ dsp2(ind1,:);
        
        spgpu2=permute(spgpu,[3 2 1]);
        matvois2=repmat(ptsgpu+spgpu2,length(rgpu),1,1)+repmat(rgpu,1,1,length(ptsgpu));
        matvoisx2=matvois2(:,1,:);
        matvoisy2=matvois2(:,2,:);
        
        count = count+1;
        ind1 = (conver(:,k)==0);
        if ~(any(ind1(:)))
            break
        end
    end
    
    ind=ldspl>threshold(k);
    
    
    
    % compute error if asked
    if nargout==3
        babs=gather(squeeze(permute(b,[3 1 2])));
        Iloc=gather(squeeze(permute(I,[3 1 2])));
        erreur(:,k) = sum(babs.^2,2)./sum(Iloc.^2,2)*100;
        
    end
    
    clear img1 img2 Px Py
end
% return values of pixels with non-integer indices 'ind' in 'array' using
% bilinear interpolation. Same purpose as 'interp2' but specific to our
% case and much quicker.

%First create a generic bilinear function and then duplicate it four time
%to apply it to either img1 img2 Px and Py
sp=double(gather(spgpu));
    function interp = bilinearPx(row, col)
        
        rowU = max(1,floor(row));  rowD = min(sx,floor(row)+1);
        colL = max(1,floor(col));  colR = min(sy,floor(col)+1);
        
        
        alpha1=col-colL;
        alpha2= row - rowU;
        beta1   = 1 - alpha1;
        beta2   = 1 - alpha2;
        
        
        
        interp =          beta1.*beta2.*Px(rowU,colL);
        interp = interp + alpha1.*beta2.*Px(rowU,colR);
        interp = interp + beta1.*alpha2.*Px(rowD,colL);
        interp = interp + alpha1.*alpha2.*Px(rowD,colR);
    end
    function interp = bilinearPy(row, col)
        
        rowU = max(1,floor(row));  rowD = min(sx,floor(row)+1);
        colL = max(1,floor(col));  colR = min(sy,floor(col)+1);
        
        
        alpha1= col-colL;
        alpha2=row - rowU;
        beta1   = 1 - alpha1;
        beta2   = 1 - alpha2;
        
        
        
        interp =          beta1.*beta2.*Py(rowU,colL);
        interp = interp + alpha1.*beta2.*Py(rowU,colR);
        interp = interp + beta1.*alpha2.*Py(rowD,colL);
        interp = interp + alpha1.*alpha2.*Py(rowD,colR);
    end

    function interp = bilinearimg1(row, col)
        
        rowU = max(1,floor(row));  rowD = min(sx,floor(row)+1);
        colL = max(1,floor(col));  colR = min(sy,floor(col)+1);
        
        
        alpha1= col-colL;
        alpha2=row - rowU;
        beta1   = 1 - alpha1;
        beta2   = 1 - alpha2;
        
        
        
        interp =          beta1.*beta2.*img1(rowU,colL);
        interp = interp + alpha1.*beta2.*img1(rowU,colR);
        interp = interp + beta1.*alpha2.*img1(rowD,colL);
        interp = interp + alpha1.*alpha2.*img1(rowD,colR);
    end
    function interp = bilinearimg2(row, col)
        
        rowU = max(1,floor(row));  rowD = min(sx,floor(row)+1);
        colL = max(1,floor(col));  colR = min(sy,floor(col)+1);
        
        
        alpha1= col-colL;
        alpha2=row - rowU;
        beta1   = 1 - alpha1;
        beta2   = 1 - alpha2;
        
        
        
        interp =          beta1.*beta2.*img2(rowU,colL);
        interp = interp + alpha1.*beta2.*img2(rowU,colR);
        interp = interp + beta1.*alpha2.*img2(rowD,colL);
        interp = interp + alpha1.*alpha2.*img2(rowD,colR);
    end


end


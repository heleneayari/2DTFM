clear
clc
close all;
%%

dossier=['Data', filesep];
nom='bead_a.tif';

info=imfinfo([dossier,nom]);
scrsz=get(groot,'ScreenSize');

%% parameters
% find local maxima
param.minDist=10;   % minimal distance in between two points
param.maxFeature=2500;  % maximal number of points
param.excludeEqualPoints=1;
% pyramid
param.pyramidLevel=3;
WinSize=24;%Size of the final interrogation window in the original image
param.blurRadius=-1; %gaussian blur of the initial image (-1 no blur)
param.rzxy=1;
param.winSize=repmat(WinSize,param.pyramidLevel,1);
% KLT
param.maxIteration=repmat(20,1,param.pyramidLevel);
param.threshold=repmat(0.1,1,param.pyramidLevel);

param.gpu=1;
%%

for ii=1
    ii
    %% read the first image
    im1=double(imread([dossier,nom],ii+1));
    im1=im1-min(im1(:));
    im1=im1/max(im1(:));
    
    %% find features
    [pts(:,1),pts(:,2)]=localMaximum_h(im1,param.minDist,param.excludeEqualPoints,param.maxFeature);
    
    %% you can also load a mask called here BW and keep only the points inside
    %[ptst(:,1),ptst(:,2)]=localMaximum_h(256-im1,param.minDist,param.excludeEqualPoints,param.maxFeature);
    
    % load([dossier,'mask.mat'])
    
    %ind=sub2ind(size(BW),ptst(:,1),ptst(:,2));
    % indf=BW(ind)==0;
    % pts=ptst(indf,:);
    %
    
    %% uncomment this part the first time you run the algorithm to check that your features are correctly selected!
    
    figure
    hold on
    imagesc(im1)
    colormap('gray')
    
    plot(pts(:,2),pts(:,1),'+g','MarkerSize',10)
    set(gca,'Ydir','reverse')
    axis equal
    axis off
    imcontrast
    %
    pause
    
    %% read second image
    im2=double(imread([dossier,nom],ii));
    im2=im2-min(im2(:));
    im2=im2/max(im2(:));
    %% calculate pyramids of image and image gradient
    pyr1=makePyramid_2D(im1,param.pyramidLevel,param.blurRadius,param.winSize);
    pyr2=makePyramid_2D(im2,param.pyramidLevel,param.blurRadius,param.winSize);
    
    
    %% KLT calculus
    if param.gpu
        [sp,conver,err]=pyrLK_2Dgpu(pyr1,pyr2,pts,param.winSize,param.maxIteration,param.threshold);
    else
        [sp,conver,err]=pyrLK_2D(pyr1,pyr2,pts,param.winSize,param.maxIteration,param.threshold);
    end
    
    
    %% save the results
    ind=prod(conver,2)>0;% features where the algorithm did converge
    res(ii).pts=pts;
    res(ii).sp=sp;
    res(ii).conver=conver;
    res(ii).ind=ind;
    sum(ind)
    save([dossier,'resklt.mat'],'res','param')
    
    
    
    %% visualisation, can be commented once everything is set

    
    
    figure('Position',[1 1 scrsz(3) scrsz(4)],'Name','klt')
   figToolbarFix
   title(['WinSize=',num2str(WinSize)])
    axis equal
    set(gca,'Ydir','reverse','Xtick',[],'Ytick',[],'XLim',[1,size(im1,2)],'Ylim',[1,size(im1,1)])
    hold on
    imagesc(im1); colormap('gray')
    %  quiverh(res(ii).pts(ind,2),res(ii).pts(ind,1),res(ii).sp(ind,2),res(ii).sp(ind,1),autoscale1/autoscalea,'r');
    quiver(res(ii).pts(ind,2), res(ii).pts(ind,1),res(ii).sp(ind,2),res(ii).sp(ind,1),'g')
    %quiver(res(ii).pts(:,2),res(ii).pts(:,1),res(ii).sp(:,2),res(ii).sp(:,1),'g')
    
    
    
    clear pts
end

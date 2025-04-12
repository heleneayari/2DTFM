clc
close all;
clear ;


eps1=0.00000001;

nu=0.499; %poisson modulus
E=40; %Young modulus
pix=0.02025;% pixel size in µm




param.E=E;
param.nu=nu;
param.pix=pix;
param.eps1=eps1;



alpha = 0.33; % Size of arrow head relative to the length of the vector
beta = 0.33;  % Width of the base of the arrow head relative to the length
uu=6;



dossier=['Data',filesep];
dossierres=[dossier];


nom='ph_a.tif';

cp=1;

for k=2:2
    
    a=imread([dossier,nom],1);
    h2=figure;
    hold on

    imagesc(a)
    colormap(gray)
    load([dossier,'resklt.mat'])
 
    
    
    %
    [Ny,Nx]=size(a);
    N=2^6;
    param.N=N;

    
    Lx=pix*Nx;
    Ly=pix*Ny;
    nm=sqrt(res.sp(:,2).^2+res.sp(:,1).^2);
    nmm=median(nm);
    

    ind=res(1).ind;
    

    Fx=TriScatteredInterp(res(1).pts(ind,2),res(1).pts(ind,1),res.sp(ind,2));
    Fy=TriScatteredInterp(res(1).pts(ind,2),res(1).pts(ind,1),res.sp(ind,1));
    
    pas=min(Nx,Ny)/N;
    [X2,Y2]=meshgrid(0:pas:(N-1)/N*Nx,0:pas:(N-1)/N*Ny);

    
    %%
    UX=Fx(X2,Y2);
    UY=Fy(X2,Y2);
    
    dimx=size(UX,2);
    dimy=size(UY,1);
    
    Fx.Method = 'nearest';
    Fy.Method= 'nearest';
    UX(isnan(UX)) = Fx(X2(isnan(UX)), Y2(isnan(UX)));
    UY(isnan(UY)) = Fy(X2(isnan(UY)), Y2(isnan(UY)));
    UX=UX*pix;
    UY=UY*pix;
    % UX=wiener2(UX,[3 3]);
    % UY=wiener2(UY,[3 3]);
    
    % UX=wiener2(UX,[12 12]);
    % UY=wiener2(UY,[12 12]);
    
    [px,sx]=perdecomp(UX);
    [py,sy]=perdecomp(UY);
    
    
    
    
    
    Y=fft2(px,size(X2,1),size(X2,2));
    Z=fft2(py,size(X2,1),size(X2,2));
    
    
    Y=fftshift(Y);
    Z=fftshift(Z);
    
    
    kx=(-(size(X2,2))*pi+eps1:2*pi:(size(X2,2)-1)*pi)/(Lx);% valable si la taille des images est paire mais en général c'est toujours le cas
    ky=(-(size(X2,1))*pi+eps1:2*pi:(size(X2,1)-1)*pi)/(Ly);% valable si la taille des images est paire mais en général c'est toujours le cas
    
    
    %         kx=(-(N-1)*pi-eps1:2*pi:(N-1)*pi-eps1)/(Lx);
    %         ky=(-(N-1)*pi-eps1:2*pi:(N-1)*pi-eps1)/(Ly);
    [KX,KY]=meshgrid(kx,ky);
    
    A=2*(1+nu)/E;
    
    K=sqrt(KX.^2+KY.^2);
    
    
    C=1./(A*(1-nu).*K);
    K1=C.*((1-nu).*K.^2+nu*KX.^2);
    K2=C.*nu.*KX.*KY;
    K3=C.*((1-nu)*K.^2+nu*KY.^2);
    
    
    VX=K1.*Y+K2.*Z;
    VY=K2.*Y+K3.*Z;
    save([dossierres,'force_fourier_',num2str(k),'.mat'],'VX','VY','param');
    
    VX=ifftshift(VX);
    VY=ifftshift(VY);
    
    
    ResultX(:,:)=real(ifft2(VX));
    ResultY(:,:)=real(ifft2(VY));
    
    
    
    figure(h2)
    
    [~,autoscalef]=quiverh(X2,Y2,ResultX(:,:),ResultY(:,:),2,'g');
    
    
    %[h,autoscalef]=quiverh(X2,Y2,UXph-nanmean(UXph(:)),UYph-nanmean(UYph(:)),2,'r');
    set(gca,'Ydir','reverse')
    %  [h,autoscalef]=quiverh(X2,Y2,UX,UY,2,'r');
    
    axis equal;
    axis tight;
    set(gca,'YDir','reverse','XTick',[],'YTick',[])
    
    ff(cp)=getframe;
    Rnorm(:,:,cp)=sqrt(real(ResultX(:,:)).^2+real(ResultY(:,:)).^2);
    
    
    figure
    imagesc(Rnorm(:,:,cp))
    
    clear res
    cp=cp+1;
    
    save([dossierres,'force_fft.mat'], 'X2','Y2','ResultX','ResultY','param')
    
    %close all
end


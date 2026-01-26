% Rough ground truth segmentation
% function that calculates ZC and Area Opening for vessel segmentation
% Input dMRA [x,y,z,t]

% Use example
% img = double(niftiread('case1.nii'));
% seg3d = ZCAO(img);

function [seg3d,imgcross] = ZCAO(img,zc_thr,prox_peak_indx,dist_peak_indx) %% takes [x,y,z,t]
tic
vox_n = size(img,1)*size(img,2)*size(img,3); %% size of 3d image
t_n = size(img,4); %% size of time domain

%% zero cross
% f = waitbar(0,'');

imgconc = reshape(img,[vox_n,t_n]); % reshape 4D to [vx,time]
t_mid = floor(size(img,4)/2); % half time index
croosconc = zeros(vox_n,1);
peak = max(imgconc(:,[prox_peak_indx dist_peak_indx]),[],2);
outerval_min = min(imgconc(:,[1:(prox_peak_indx-3) (dist_peak_indx+3):end]),[],2);

parfor i = 1:vox_n 
    if (peak(i)>0) && (peak(i)>outerval_min(i)) && (peak(i)>imgconc(i,1)) && (peak(i)>imgconc(i,end)) % Rule based conditions to remove non-vessels
        [rate, n] = zerocrossrate(imgconc(i,:),Level=mean(imgconc(i,:)),Method="comparison",Threshold=zc_thr);
    else
        n = 0;
    end
    croosconc(i) = n; % number of zero crossings
end

imgcross = reshape(croosconc,[size(img,1) size(img,2) size(img,3)]); % reshape back to 3D
imgcross(imgcross<=1) = 0; % another condition to remove bone
imgcross(isnan(imgcross)) = 0; % 
imgcross(imgcross==inf) = 0; % 

%% area opening code
seg3d = bwareaopen(imgcross,3);

toc
end %% function end

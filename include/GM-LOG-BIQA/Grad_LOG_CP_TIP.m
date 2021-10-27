function out = Grad_LOG_CP_TIP(imd)

% Grad_LOG_CP_TIP - measure the distortion degree of distorted image 'imd'
% based on the statistical properties of Gradient Magnitude and LOG
% resposne.

% inputs:
% imd - the distorted image (grayscale image, double type, 0~255)

% outputs:
% out(1:20): the two marginal distribution for GM and LOG response. 
% out(21:40): the independency distributions between GM and LOG response. 

% This is an implementation of the NR IQA algorithm in the following paper:
%  W. Xue, X. Mou, L. Zhang, Alan C. Bovik, and X. Feng, ¡°Blind Image Quality Prediction Using
% Joint Statistics of Gradient Magnitude and Laplacian Features,¡± submitted to 
% Trans. on Image Processing, IEEE.

sigma = 0.5;
[gx,gy] = gaussian_derivative(imd,sigma);
grad_im = sqrt(gx.^2+gy.^2);

window2 = fspecial('log', 2*ceil(3*sigma)+1, sigma);
window2 =  window2/sum(abs(window2(:)));
log_im = abs(filter2(window2, imd, 'same'));

ratio = 2.5; % default value 2.5 is the average ratio of GM to LOG on LIVE database
grad_im = abs(grad_im/ratio);

%Normalization
c0 = 4*0.05;
sigmaN = 2*sigma;
window1 = fspecial('gaussian',2*ceil(3*sigmaN)+1, sigmaN);
window1 = window1/sum(window1(:));
Nmap = sqrt(filter2(window1,mean(cat(3,grad_im,log_im).^2,3),'same'))+c0;
grad_im = (grad_im)./Nmap;
log_im = (log_im)./Nmap;
% remove the borders, which may be the wrong results of a convolution
% operation
h = ceil(3*sigmaN);
grad_im = abs(grad_im(h:end-h+1,h:end-h+1,:));
log_im = abs(log_im(h:end-h+1,h:end-h+1));

ctrs{1} = 1:10;ctrs{2} = 1:10;
% histogram computation
step1 = 0.20;
step2 = 0.20;
grad_qun = ceil(grad_im/step1);
log_im_qun = ceil(log_im/step2);

N1 = hist3([grad_qun(:),log_im_qun(:)],ctrs);
N1 = N1/sum(N1(:));
NG = sum(N1,2); NL = sum(N1,1);

alpha1 = 0.0001;
% condition probability: Grad conditioned on LOG
cp_GL = N1./(repmat(NL,size(N1,1),1)+alpha1);
cp_GL_H=  sum(cp_GL,2)';
cp_GL_H = cp_GL_H/sum(cp_GL_H);
% condition probability: LOG conditioned on Grad
cp_LG = N1./(repmat(NG,1,size(N1,2))+alpha1);
cp_LG_H = sum(cp_LG,1);
cp_LG_H = cp_LG_H/(sum(cp_LG_H));

out = [NG', NL, cp_GL_H,cp_LG_H];


function [gx,gy] = gaussian_derivative(imd,sigma)
window1 = fspecial('gaussian',2*ceil(3*sigma)+1+2, sigma);
winx = window1(2:end-1,2:end-1)-window1(2:end-1,3:end);winx = winx/sum(abs(winx(:)));
winy = window1(2:end-1,2:end-1)-window1(3:end,2:end-1);winy = winy/sum(abs(winy(:)));
gx = filter2(winx,imd,'same');
gy = filter2(winy,imd,'same');
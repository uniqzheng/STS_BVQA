
 load modelparameters.mat
 
 blocksizerow    = 96;
 blocksizecol    = 96;
 blockrowoverlap = 0;
 blockcoloverlap = 0;

im =imread('image1.bmp');

quality1 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
    mu_prisparam,cov_prisparam)

im =imread('image2.bmp');

quality2 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
    mu_prisparam,cov_prisparam)

im =imread('image3.bmp');

quality3 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
    mu_prisparam,cov_prisparam)

im =imread('image4.bmp');

quality4 = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
    mu_prisparam,cov_prisparam)
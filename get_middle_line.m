function get_middle_line()
%GET_MIDDLE_LINE Summary of this function goes here
%   Detailed explanation goes here
close all;
img = imread("data/000127_mask.jpg");
figure, imshow(img, []);
img = double(img);
% [height, width] = size(img);
mask = img == 255;

rowsum = sum(mask, 2);
meansum = mean(rowsum(:));
thr = 0.98;

radius = floor(meansum/3);
pmask = padarray(mask, [radius, radius], 'replicate', 'both');

kernel = fspecial('disk', radius);
fimg = filter2(kernel, pmask, 'valid');
fmask = fimg > thr;

pfmask = padarray(fmask, [radius, radius], 'replicate', 'both');
pskel = bwmorph(pfmask, 'skeleton', Inf);
skel = pskel(radius+1:end-radius, radius+1:end-radius);
figure, imshow(skel, []);
end


% The investigators in this research are: 
% Debarati Kundu ( debarati@utexas.edu ) -- Graduate student, Dept. of ECE, UT Austin.
% Deepti Ghadiyaram ( deeptigp9@gmail.com ) -- Graduate student, Dept. of CS, UT Austin.
% Dr. Alan C. Bovik ( bovik@ece.utexas.edu ) -- Professor, Dept. of ECE, UT Austin
% Dr. Brian L. Evans ( bevans@ece.utexas.edu ) -- Professor, Dept. of ECE, UT Austin
% Kindly report any suggestions or corrections to Debarati Kundu (debarati@utexas.edu)
%
% This function computes the features of the HIGRADE-1 model.

function feat = higrade_1(imHDR)
	cform = makecform('srgb2lab');
	imHDR_lab = applycform(imHDR, cform);
	feat = [higrade_grad_log(imHDR_lab(:,:,1)) higrade_grad_log(imHDR_lab(:,:,2)) higrade_grad_log(imHDR_lab(:,:,3))];
end

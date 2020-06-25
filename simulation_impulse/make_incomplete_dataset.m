clear; clc; close all;

%%
load('VIB.mat');

rng(3);
loss_ratio = 0.2;

num_row = size(VIB, 1);
num_col = size(VIB, 2);
num_row_missing = floor(num_row * loss_ratio);

mask_matrix = ones(num_row, num_col);

for c = 1 : num_col
    mask_matrix(randsample(num_row, num_row_missing), c) = NaN;    
end

VIB = VIB .* mask_matrix;

%%
load('DPM.mat');

rng(4);
loss_ratio = 0.2;

num_row = size(DPM, 1);
num_col = size(DPM, 2);
num_row_missing = floor(num_row * loss_ratio);

mask_matrix = ones(num_row, num_col);

for c = 1 : num_col
    mask_matrix(randsample(num_row, num_row_missing), c) = NaN;    
end

DPM = DPM .* mask_matrix;






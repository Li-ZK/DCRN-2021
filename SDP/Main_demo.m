% This code is for our paper "Spatial Density Peak Clustering for 
% Hyperspectral Image Classification with Noisy Label, Accpeted, 2019".

% If you have any questions, please contact us. 
% Email: tubing@hnist.edu.cn; xiaofei_zh@foxmail.com and
% xudong_kang@163.com

clc;clear;close all;
lambda = 0.01;scale = 1;K=2;
C = 13;
load KSC;load KSC_gt;
tim2=(KSC_gt);
tim1=double(tim2);
[xi,yi] =find(tim1==0);
xisize=size(xi);
[r,c,b] = size(img);
GroundT = GroundT';
img_2 = reshape(img,[r*c,b]);
img_3 = double(img_2);

indexes = train_test_random_Value(GroundT(2,:));
train_SL = GroundT(:,indexes);
train_samples = img_3(train_SL(1,:),:);
Test_SL = GroundT;
Test_SL(:,indexes) = [];
Test_SL_samples = img_3(Test_SL(1,:),:);
train_labels_C= train_SL(2,:)';
per = 9;
[train_labels_noise_1,noise_train_data,noise_train_data1] = LNA_fei(...
    train_SL,Test_SL,train_labels_C,train_samples,Test_SL_samples,per);

train_labels_noise = [noise_train_data1(:,2)';train_labels_noise_1'];
for i = 1:size(train_labels_noise,2)
    train_sample_error(i) = find(GroundT(1,:)==train_labels_noise(1,i));
end
test = GroundT;
test(:,train_sample_error) = [];
test_sample = test(1,:); test_label = test(2,:); 
test_data_ori = img_2(test_sample,:);i_row = r;  i_col=c;
tt_index = train_labels_noise(1,:);
index_map = reshape(1:size(img_2,1),[i_row,i_col]);
train_joint = [];
KSC_gt(train_labels_noise(1,:)) = train_labels_noise(2,:);

[train_joint,train_joint_label,train_joint_data] = my_train_joint(i_row,...
    i_col,scale,GroundT,tt_index,KSC_gt,index_map,img_3);

for i = 1:C
    posi = find(train_labels_noise(2,:)==i);
    label_kind{i} = train_labels_noise(1,posi);
    label_col = ceil(label_kind{i}/r);          
    label_row = mod(label_kind{i},r);
    label_spectra_1{i} = [posi;label_kind{i};label_row;label_col;...
        img_3(label_kind{i},:)'];
    label_spectra_2{i} = (label_spectra_1{i}((5:end),:));
end

[train_sample_correct,~] = my_dp_average_K(label_spectra_1,label_spectra_2,...
    train_joint_data,lambda,K);
train_label_correct=KSC_gt(train_sample_correct);

[OA,AA,kappa,CA,result] = my_correct_accury(train_label_correct,img,...
    GroundT,train_sample_correct,train_labels_noise,xi,yi);
result = reshape(result,r,c);
VClassMap=label2color_ksc(result,'ksc');
figure()
imshow(VClassMap);
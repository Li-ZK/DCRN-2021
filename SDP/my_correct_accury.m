function [OA,AA,kappa,CA,Result] = my_correct_accury(train_label_correct,fimage,GroundT,train_sample_correct,train_labels_noise,xi,yi);

[r,c,~] = size(fimage);
train_sample_correct_1 = zeros(1,size(train_sample_correct,2));
for i = 1:size(train_sample_correct,2)
    train_sample_correct_1(i) = find(GroundT(1,:)==train_sample_correct(i));
end
test = GroundT;
test(:,train_sample_correct_1) = [];
test_sample = test(1,:);
test_label = test(2,:);

%% ======================矫正label_noise的分类结果==========================
fimg = reshape(fimage,r*c,size(fimage,3));
train_samples = fimg(train_sample_correct,:);
train_sample_err = fimg(train_labels_noise(1,:),:);
[train_samples,M,m] = scale_func(train_samples);
[train_sample_err ] = scale_func(train_sample_err,M,m);
[fimg ] = scale_func(fimg,M,m);
[Ccv Gcv cv cv_t]=cross_validation_svm(train_label_correct',train_samples);
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
model=svmtrain(train_label_correct',train_samples,parameter);
Result = svmpredict(ones(r*c,1),fimg,model);
GroudTest = double(test_label);
ResultTest = Result(test_sample,:);
[OA,AA,kappa,CA] = confusion(GroudTest,ResultTest);
end
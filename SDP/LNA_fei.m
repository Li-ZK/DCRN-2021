function [noise_train_label,noise_train_data,noise_train_data1] = LNA_fei(train_SL,Test_SL,train_labels,train_samples,Test_SL_samples,per)
PretreatmentLabel = [];
NoiseLD = [];
NoiseData = [];
noise_train_data1 = [];
noise_train_label = [];
LabelData =[(train_SL(2,:))',(train_SL(1,:))',train_samples]; 
MiddleData = LabelData;
for i = 1:max(train_labels)
    if i>1
       MiddleData(find(MiddleData(:,1) == i-1),:) = []; 
    end
    LabelCount(i) = length(find(train_labels == i));
    NoiseCount(i) = per;
    PretreatmentLabel = [PretreatmentLabel;i*ones(NoiseCount(i),1)];  
    train_labels_1 = ones(1,LabelCount(i)+per)*i;
    noise_train_label = [noise_train_label train_labels_1];
end
NoiseLD = [(Test_SL(2,:))' (Test_SL(1,:))' Test_SL_samples];
for j = 1:max(PretreatmentLabel)
    NoiseCount1 = NoiseCount;
    NoiseLD1 = NoiseLD;
    NoiseCount1(j) = [];
    NoiseLD1(find(NoiseLD1(:,1) == j),:) = [];
    AfterLabel = randi([1,size(NoiseLD1,1)],1,NoiseCount(j));
    NoiseDataMiddle = NoiseLD1(AfterLabel,:);
    NoiseData = [NoiseData;NoiseDataMiddle];
end
NoiseData_Bfor = [PretreatmentLabel,NoiseData];
for k = 1:max(PretreatmentLabel)
    Pdata = LabelData(find(LabelData(:,1)==k),:);
    Ndata = NoiseData_Bfor(find(NoiseData_Bfor(:,1)==k),:);
    Edata = [Ndata(:,(2:end));Pdata];  
    noise_train_data1 = [noise_train_data1;Edata];
end
noise_train_data = noise_train_data1(:,3:end);
noise_train_label = noise_train_label';
end
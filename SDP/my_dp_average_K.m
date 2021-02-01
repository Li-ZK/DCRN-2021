function [train_sample_correct,xxxxxx] = my_dp_average_K(label_spectra_1,label_spectra_2,train_joint_data,lambda,K)
C = size(label_spectra_2,2);
para.method = 'gaussian';
para.percent = 26.0;
label_spectra_4 = label_spectra_1;
label_spectra_3 = [];train_sample_correct = [];xxxxxx = [];
for i = 1:C
    d_1 = [];
    for z = 1: size(label_spectra_2{i},2)
        label_spectra_posi_1 = label_spectra_1{i}(1,:);
        label_joint_data_1 = train_joint_data(label_spectra_posi_1);
        for zx = 1:size(label_joint_data_1{z},1)
            train_joint_1 = (repmat(label_joint_data_1{z}(zx,:),size(label_spectra_2{i},2),1))';
            for j = 1:size(train_joint_1,2)
                A = corr2(train_joint_1(:,j),label_spectra_2{i}(:,j));
                d_1{z}(j,zx) = 1-A;
            end
        end
    end

    for z = 1:size(d_1,2)
        k = K;
        xxx  =size(d_1{z},1);
            if xxx<k
                k=xxx;
            else
                k = K;
            end
            sort_d = [];
        for zx = 1:size(d_1{z},2)
            sort_d = sort(d_1{z}(:,zx),'descend');
            d_3 = sort_d(1:k);
            label_spectra_3{i}(z,zx) = sum(d_3)/k;
        end

    end
    [rho] = cluster_dp_auto(label_spectra_3{i}, para);
    rho_1_mean = mean(rho);
    rho_1_limit = lambda * rho_1_mean;
    xxx = find(rho<rho_1_limit);
    label_spectra_4{i}(:,xxx)=[];
    train_sample_correct = [train_sample_correct label_spectra_4{i}(2,:)]; 
    xxxxxx = [xxxxxx xxx+(size(train_joint_data,2)/C)*(i-1)];

end
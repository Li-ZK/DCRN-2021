function train_sample_correct = my_dp_average_w_K(label_spectra_1,label_spectra_2,train_joint_data,lambda,K,a,b,c)
C = size(label_spectra_2,2);
para.method = 'gaussian';
para.percent = 15.0;
label_spectra_4 = label_spectra_1;
label_spectra_3 = [];train_sample_correct = [];
for i = 1:C
    d_1 = [];
    for z = 1: size(label_spectra_2{i},2);
        label_spectra_posi_1 = label_spectra_1{i}(1,:);
        label_joint_data_1 = train_joint_data(label_spectra_posi_1);
        for zx = 1:size(label_joint_data_1{z},1);
            train_joint_1 = (repmat(label_joint_data_1{z}(zx,:),size(label_spectra_2{i},2),1))';
            d_1{z}(zx,:) = sqrt(sum((train_joint_1 - label_spectra_2{i}).^2));  
        end
    end

    for z = 1:size(d_1,2)
        k = K;
        xxx  =size(d_1{z},1);
            if xxx<k
                k=xxx;
            else
                k = 5;
            end
            sort_d = [];
        for zx = 1:size(d_1{z},2)
            sort_d = sort(d_1{z}(:,zx),'descend');
            d_3 = sort_d(1:k);
            wd_gaussian = a * exp(-(d_3 -b).^2 /(2 * c.^2));
            W_dist =d_3.*wd_gaussian;   % w系数
            label_spectra_3{i}(z,zx) =sum(W_dist)/sum(wd_gaussian(1:end));  %加权平均
%             label_spectra_3{i}(z,zx) = sum(d_3)/K;
        end
    end
    [rho] = cluster_dp_auto(label_spectra_3{i}, para);
    rho_1_mean = mean(rho);
    rho_1_limit = lambda * rho_1_mean;
    xxx = find(rho<rho_1_limit);
    label_spectra_4{i}(:,xxx)=[];
    train_sample_correct = [train_sample_correct label_spectra_4{i}(2,:)]; 
end
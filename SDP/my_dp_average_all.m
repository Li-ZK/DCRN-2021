function train_sample_correct = my_dp_average_all(label_spectra_1,label_spectra_2,train_joint_data,lambda)
    para.method = 'gaussian';
    para.percent = 15.0;
    label_spectra_4 = label_spectra_1;
    label_spectra_3 = [];train_sample_correct = [];
    C = size(label_spectra_2,2);
    for i = 1:C
        d = [];label_spectra_posi_1 = label_spectra_1{i}(1,:);
        for z = 1: size(label_spectra_2{i},2);
            label_joint_data_1 = train_joint_data(label_spectra_posi_1);
            for zx = 1:size(label_joint_data_1{z},1);
                train_joint_1 = (repmat(label_joint_data_1{z}(zx,:),size(label_spectra_2{i},2),1))';
                for j = 1:size(train_joint_1,2)
%                 d(zx,:) = sqrt(sum((train_joint_1 - label_spectra_2{i}).^2));  
                d(zx,j) = corr2(train_joint_1(:,j),label_spectra_2{i}(:,j));
                end
            end
            label_spectra_3{i}(z,:) = mean(1-d,1);
        end
        
%                 for z = 1: size(label_spectra_2{i},2);
%                     for j = 1:size(label_spectra_2{i},2);
%                         A = corr2(label_spectra_2{i}(:,z),label_spectra_2{i}(:,j));
%                         label_spectra_3{i}(z,j) = 1-A;
%                     end
%                 end
        
        [rho] = cluster_dp_auto(label_spectra_3{i}, para);
        rho_1_mean = mean(rho);
        rho_1_limit = lambda * rho_1_mean;
        xxx = find(rho<rho_1_limit);
        label_spectra_4{i}(:,xxx)=[];
        train_sample_correct = [train_sample_correct label_spectra_4{i}(2,:)]; 
%         train_label_correct
    end
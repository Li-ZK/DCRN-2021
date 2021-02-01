function [train_joint,train_joint_label,train_joint_data] = my_train_joint(i_row,i_col,scale,GroundT,tt_index,indian_pines_gt,index_map,img3)
    for p = 1:size(tt_index,2)
        row = mod(tt_index(p),i_row);  
        if row == 0
            row = i_row;
        end
        col = ceil(tt_index(p)/i_row); 
        row_range = row-scale : row+scale;
        row_range(row_range<=0)= 1;row_range(row_range>=i_row)= i_row; 
        col_range = col-scale : col+scale;
        col_range(col_range<=0)= 1;col_range(col_range>=i_col)= i_col; 
        y_jonit_index = index_map(row_range,col_range);
        y_jonit_index_row = reshape(y_jonit_index,1,length(row_range)*length(col_range));
        [y_intersect,tt_index_location,y_jonit_index_row_location] = intersect(GroundT(1,:) , y_jonit_index_row); 
        train_joint{p} = y_intersect;
        train_joint_label{p} = indian_pines_gt(train_joint{p});
        train_joint_data{p} = img3(train_joint{p},:);
    end
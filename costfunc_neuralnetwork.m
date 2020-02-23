function [Jval,Dvec] = costfunc_neuralnetwork(thetavec,x,y)

[j,~] = size(x);

%unroll theta here
theta1 = reshape(thetavec(1:30),5,6);
theta2 = reshape(thetavec(31:60),5,6);
theta3 = reshape(thetavec(61:90),5,6);
theta4 = reshape(thetavec(91:108),3,6);

Jval = 0;
total_cost = 0;

grad_1 = 0;
grad_2 = 0;
grad_3 = 0;
grad_4 = 0;

no_elm = max(y);

for i = 1:j
    %% FORWARD PROPAGATION
        
    a1 = x(i,:);
    a1_ = ones(1,6);
    a1_(2:6) = a1;
    
    a2 = sigmoid_nn(a1_,theta1);
    a2_ = ones(1,6);
    a2_(2:6) = a2';
    
    a3 = sigmoid_nn(a2_,theta2);
    a3_ = ones(1,6);
    a3_(2:6) = a3';
    
    a4 = sigmoid_nn(a3_,theta3);
    a4_ = ones(1,6);
    a4_(2:6) = a4';
    
    a5 = sigmoid_nn(a4_,theta4);
    a5_ = ones(1,3);
    a5_(1:3) = a5';
    
    h0 = a5_;
    %as a5 layer is the last layer
    
    %% SUMMATION OF COST Jval
    
    %setting y as a matrix
    
    class_y = multi_class(y(i),no_elm);
    
    %to get Jval
    
    [~,ab] = size(h0);
        
    for ind = 1:ab
        
        yi = class_y(ind);
        h0i = h0(ind);
        if yi == 0 && h0i <= 1e-6
            cost = 0;
        elseif 1-yi ==0 && 1-h0i <= 1e-6
            cost = 0;
        else
            cost = (yi*log(h0i)) + ((1-yi)*log(1-h0i)); 
        end
        total_cost = total_cost + cost;
        
    end
     
    %% BACK PROPAGATION
    
    error_5 = h0' - class_y';
    
    error_4 = ((theta4')*(error_5)) .* a4_' .* (1-a4_');
    error_4 = error_4(2:6);
       
    error_3 = ((theta3')*(error_4)) .* a3_' .* (1-a3_');
    error_3 = error_3(2:6);
    
    error_2 = ((theta2')*(error_3)) .* a2_' .* (1-a2_');
    error_2 = error_2(2:6);
    
    grad_4 = grad_4 + error_5 * a4_;
    grad_3 = grad_3 + error_4 * a3_;
    grad_2 = grad_2 + error_3 * a2_;
    grad_1 = grad_1 + error_2 * a1_;
        
end    

%finalizing Jval
Jval = (-1/j) * total_cost;

%specifying D vector
D4 = (1/j) * grad_4;
D3 = (1/j) * grad_3;
D2 = (1/j) * grad_2;
D1 = (1/j) * grad_1;

Dvec = [D1(:)' D2(:)' D3(:)' D4(:)'];

end
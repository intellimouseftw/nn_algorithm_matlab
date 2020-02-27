function val = sigmoid_nn(x,theta)

z = theta * x';
val = 1./(1+exp((-1)*z));

end
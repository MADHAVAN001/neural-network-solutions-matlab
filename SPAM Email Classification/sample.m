P_train = P_train_std; Val.P = Val_std.P; %%% Use this line if you use STD preprocessing on the data. IMPORTANT: Run preprocess.m first

num_iterations = 10;
Missclassification_rate = zeros(num_iterations,1);
mse_performance = zeros(num_iterations,1);
num_hidden_layers = 1;

[net] = newff(minmax(P_train),[10 10 1],{'tansig','tansig','tansig'},'traingd');
net.trainParam.epochs =200;
net.trainParam.min_grad=1e-20;
net.trainParam.lr = 0.51;
net.trainParam.max_fail = 25;
net.performFcn = 'mse';  % Mean squared error

net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 30/100;
net.divideParam.testRatio = 0/100;


%train
[net,tr] = train(net,P_train,T_train,[],[],Val);
[fields,N] = size(T_test);
neuralnetscore = sign(sim(net,Val.P));

outputs = net(P_train);
errors = gsubtract(T_train,outputs);

%test
test_inputs = Val.P;
test_outputs = net(test_inputs);


Missclassification_rate = sum(0.5*abs(T_test - neuralnetscore))/N
%mse_performance((i-10)/10+1,1) = perform(net,T_test,test_outputs);

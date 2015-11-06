P_train = P_train_std; Val.P = Val_std.P; %%% Use this line if you use STD preprocessing on the data. IMPORTANT: Run preprocess.m first 
net = newff(minmax(P_train),[10 1],{'tansig','tansig'},'trainlm'); 
net.trainParam.epochs =100;
net.trainParam.max_fail = 25;
[net tr] = train(net,P_train,T_train,[],[],Val);
[fields N] = size(T_test);
neuralnetscore = sign(sim(net,Val.P));
Missclassification_rate = sum(0.5*abs(T_test - neuralnetscore))/N

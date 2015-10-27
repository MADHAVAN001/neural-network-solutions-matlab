%P_train = P_train_n; T_train = T_train_n; Val = Val_n; %%% Use this line to use mnmx preprocessing on the data. IMPORTANT: Run preprocess.m first 
P_train = P_train_std; T_train = T_train_std; Val = Val_std; %%% Use this line to use STD preprocessing on the data. IMPORTANT: Run preprocess.m first 

net = newff(minmax(P_train),[10 1],{'tansig','tansig'},'traingd');
net.trainParam.epochs =200;
net.trainParam.max_fail = 50;
[net tr] = train(net,P_train,T_train,[],[],Val);
[fields N] = size(T_test);

est = sim(net,Val.P);
%est = postmnmx(est,mint,maxt); %%% Use this line if you use mnmx preprocessing on the data. IMPORTANT: Uncomment the corresponding line above
est = poststd(est,meant,stdt); %%% Use this line if you use STD or PCA preprocessing on the data. IMPORTANT: Uncomment the corresponding line above 

RMS_Error = sqrt(mean((T_test - est).^2))

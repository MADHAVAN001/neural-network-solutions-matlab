P_train = P_train_std; T_train = T_train_std; Val = Val_std; %%% Use this line to use STD preprocessing on the data. IMPORTANT: Run preprocess.m first
for i = 50:50:400
    net = newrb(P_train,T_train,0,1,i,50);
    %net.trainParam.epochs =100;
    %net.trainParam.max_fail = 50;
    [fields N] = size(T_test);
    
    est = sim(net,Val.P);
    %est = postmnmx(est,mint,maxt); %%% Use this line if you use mnmx preprocessing on the data. IMPORTANT: Uncomment the corresponding line above
    est = poststd(est,meant,stdt); %%% Use this line if you use STD or PCA preprocessing on the data. IMPORTANT: Uncomment the corresponding line above
    
    RMS_Error = sqrt(mean((T_test - est).^2))
end
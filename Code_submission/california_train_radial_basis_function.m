P_train = P_train_std; T_train = T_train_std; Val = Val_std; %%% Use this line to use STD preprocessing on the data. IMPORTANT: Run preprocess.m first
for i = 50:50:400
    net = newrb(P_train,T_train,0,1,i,50);
    [fields N] = size(T_test);
    
    est = sim(net,Val.P);
    est = poststd(est,meant,stdt); 
    
    RMS_Error = sqrt(mean((T_test - est).^2))
end
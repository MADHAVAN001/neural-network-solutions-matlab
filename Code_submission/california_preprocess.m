[P_train_std meanp stdp] = prestd(P_train);
Val_std.P = trastd(P_test,meanp,stdp);
[T_train_std meant stdt] = prestd(T_train);
Val_std.T = trastd(T_test,meant,stdt);

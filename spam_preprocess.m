[P_train_std meanp stdp] = prestd(P_train);
Val_std.P = trastd(P_test,meanp,stdp);
Val_std.T = T_test;

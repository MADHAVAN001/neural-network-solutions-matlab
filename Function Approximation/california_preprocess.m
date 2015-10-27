Val.P = P_test; Val.T = T_test;
[P_train_n minp maxp] = premnmx(P_train);
Val_n.P = tramnmx(P_test,minp,maxp);
[T_train_n mint maxt] = premnmx(T_train);
Val_n.T = tramnmx(T_test,mint,maxt);

[P_train_std meanp stdp] = prestd(P_train);
Val_std.P = trastd(P_test,meanp,stdp);
[T_train_std meant stdt] = prestd(T_train);
Val_std.T = trastd(T_test,meant,stdt);

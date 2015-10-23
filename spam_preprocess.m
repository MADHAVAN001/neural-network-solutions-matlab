Val.P = P_test; Val.T = T_test;
[P_train_n minp maxp] = premnmx(P_train);
Val_n.P = tramnmx(P_test,minp,maxp);
Val_n.T = T_test;
[P_train_std meanp stdp] = prestd(P_train);
Val_std.P = trastd(P_test,meanp,stdp);
Val_std.T = T_test;

P_train = P_train_std; Val.P = Val_std.P; %%% Use this line if you use STD preprocessing on the data. IMPORTANT: Run preprocess.m first

num_iterations = 5;
Missclassification_rate = zeros(num_iterations,1);
mse_performance = zeros(num_iterations,1);
num_hidden_layers = 1;
num_neurons_third = 10;

for num_neurons = 10:10:50
    for i = 1:100
        [net] = newff(minmax(P_train),[num_neurons num_neurons_second num_neurons_third 1],{'tansig','tansig','tansig','tansig'},'trainlm');
        net.trainParam.epochs =200;
        net.trainParam.min_grad=1e-20;
        net.trainParam.max_fail = i;
        %net.trainParam.mu_max = 1e+20;
        %net.trainParam.mu_min = 1e-200;
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
        est = sim(net,Val.P);
        est = poststd(est,meant,stdt); %%% Use this line if you use STD or PCA preprocessing on the data. IMPORTANT: Uncomment the corresponding line above
        
        Missclassification_rate(i,1) = sqrt(mean((T_test - est).^2));
        mse_performance(i/i,1) = perform(net,T_test,est);
        
        %plot and save the performance figure
        name = ['Trainlm_varymaxfail\performance_maxfail',num2str(i),'_',num2str(num_neurons_second),'_',num2str(num_hidden_layers),'_',num2str(num_neurons)];
        h=figure;
        plotperform(tr);
        saveas(h,name,'jpg');
        
        %plot and save trainstate
        name = ['Trainlm_varymaxfail\plottrainstate',num2str(num_neurons_second),'_',num2str(num_hidden_layers),'_',num2str(num_neurons)];
        h=figure;
        plottrainstate(tr);
        saveas(h,name,'jpg');
        
        %plot and save regression
        name = ['Trainlm_varymaxfail\plotregression',num2str(num_neurons_second),'_',num2str(num_hidden_layers),'_',num2str(num_neurons)];
        h=figure;
        plotregression(T_train,outputs);
        saveas(h,name,'jpg');
        
        
        %plot and save histogram
        name = ['Trainlm_varymaxfail\plothistogram',num2str(num_neurons_second),'_',num2str(num_hidden_layers),'_',num2str(num_neurons)];
        h=figure;
        ploterrhist(errors);
        saveas(h,name,'jpg');
        
    end
    x = zeros(num_iterations,1);
    for i = 1:num_iterations
        x(i,1) = i;
    end
    
    %plotting misclassification rate
    name = ['Trainlm_varymaxfail\misclassification_rate',num2str(num_neurons_second)];
    h=figure;
    plot(x,Missclassification_rate);
    saveas(h,name,'jpg');
    
    
    %plotting misclassification rate
    h=figure;
    plot(x,mse_performance);
    name = ['Trainlm_varymaxfail\mseperformance_rate',num2str(num_neurons_second)];
    saveas(h,name,'jpg');
end

%save('Misclassification_rate',Missclassification_rate);
P_train = P_train_std; Val.P = Val_std.P; %%% Use this line if you use STD preprocessing on the data. IMPORTANT: Run preprocess.m first

num_iterations = 5;
Missclassification_rate = zeros(num_iterations,1);
mse_performance = zeros(num_iterations,1);
num_hidden_layers = 1;
for num_neurons_second = 10:10:10
    for alpha = 0.01:0.05:0.55 
    for num_neurons = 10:10:50
        [net] = newff(minmax(P_train),[num_neurons 10 1],{'tansig','tansig','tansig'},'traingd');
        net.trainParam.epochs =200;
        net.trainParam.min_grad=1e-20;
        net.trainParam.max_fail = 25;
        net.trainParam.lr = alpha;
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
        
        
        Missclassification_rate((num_neurons-10)/10+1,1) = sum(0.5*abs(T_test - neuralnetscore))/N;
        mse_performance((num_neurons-10)/10+1,1) = perform(net,T_test,test_outputs);
        
        %plot and save the performance figure
        name = ['Traingd5\performance_',num2str(num_neurons),'_',num2str(num_neurons_second),'_',num2str(alpha*100)];
        h=figure;
        plotperform(tr);
        saveas(h,name,'jpg');
        
        %plot and save trainstate
        name = ['Traingd5\plottrainstate',num2str(num_neurons),'_',num2str(num_neurons_second),'_',num2str(alpha*100)];
        h=figure;
        plottrainstate(tr);
        saveas(h,name,'jpg');
        
        %plot and save regression
        name = ['Traingd5\plotregression',num2str(num_neurons),'_',num2str(num_neurons_second),'_',num2str(alpha*100)];
        h=figure;
        plotregression(T_train,outputs);
        saveas(h,name,'jpg');
        
        
        %plot and save histogram
        name = ['Traingd5\plothistogram',num2str(num_neurons),'_',num2str(num_neurons_second),'_',num2str(alpha*100)];
        h=figure;
        ploterrhist(errors);
        saveas(h,name,'jpg');
        
    end
    x = zeros(num_iterations,1);
    for i = 1:num_iterations
        x(i,1) = i;
    end
    
    %plotting misclassification rate
    name = ['Traingd5\misclassification_rate',num2str(alpha*100)];
    h=figure;
    plot(x,Missclassification_rate);
    saveas(h,name,'jpg');
    
    
    %plotting misclassification rate
    h=figure;
    plot(x,mse_performance);
    name = ['Traingd5\mseperformance_rate',num2str(alpha*100)];
    saveas(h,name,'jpg');
    
    
    name = ['Traingd5\Misclassification_rate',num2str(alpha*100)];
    save(name);
    end
end

clear;
clc;
warning off;
addpath(genpath('./'));

%% ================= Excel 初始化（彻底正确版） =================
excel_name = 'result.xlsx';

% 如果文件已存在，先删除（防止重复 append）
if exist(excel_name,'file')
    delete(excel_name);
end

varNames = { ...
    'miss_rate','m','d','gamma','alpha','beta', ...
    'ACC','NMI','Purity','Fscore', ...
    'ACC_std','NMI_std','Purity_std','Fscore_std', ...
    'IterNum','Time'};

% 主结果表（只有表头，没有数据）
T_all = table('Size',[0,numel(varNames)], ...
    'VariableTypes', repmat("double",1,numel(varNames)), ...
    'VariableNames', varNames);
writetable(T_all, excel_name);

% Best_ACC 表（只有表头）
T_best = table('Size',[0,numel(varNames)], ...
    'VariableTypes', repmat("double",1,numel(varNames)), ...
    'VariableNames', varNames);
writetable(T_best, excel_name, 'Sheet', 'Best_ACC');

%% ================= Dataset =================
ds = {'MSRC-v1'};

i_fold = 1;
num_cluster_runs = 20; % Repeat final clustering on the same learned F.

for dsi = 1:length(ds)

    %% ---------- load data ----------
    dataName = ds{dsi};
    load(dataName);
    % % Y=y;
    % X=data;
    % Y=gt;
   

  param.c   = length(unique(Y));
    nCluster  = param.c;
    n         = length(Y);
    num_view  = length(X);

    original_X = X;
    original_Y = Y;

    miss_rates = [0.1 0.3 0.5 0.7];
  

    %% ================= miss_rate 循环 =================
    for i_miss = 1:length(miss_rates)

        % ---------- best 初始化 ----------
        best_ACC   = -inf;
        best_res   = [];
        best_std   = [];
        best_param = struct();
        best_iter  = 0;
        best_time  = 0;

        % ---------- 恢复原始数据 ----------
        X = original_X;
        Y = original_Y;

        mr = miss_rates(i_miss);

        %% ---------- load missing index ----------
        foldName = strcat(dataName,'_percentDel_',num2str(mr),'.mat');
        load(foldName);
        ind_folds = folds{i_fold};

        %% ---------- construct ind_0 ind_1 ----------
        ind_0 = cell(1,num_view);
        ind_1 = cell(1,num_view);
        for iv = 1:num_view            
            ind_0{iv} = find(ind_folds(:,iv) == 0);
            ind_1{iv} = find(ind_folds(:,iv) == 1);
        end
        for v = 1:num_view
            idx = ind_1{v};
            n_v = length(idx);

            Tv = zeros(n, n_v);
             for j = 1:n_v
                   Tv(idx(j), j) = 1;
             end
            TIND{v} = Tv;
        end

        
        %% ---------- initialize X ----------
        for iv = 1:num_view
            if size(X{iv},2) ~= n
                X{iv} = X{iv}';
            end
            X{iv} = NormalizeFea(X{iv}, 0);
            X{iv}(:,ind_0{iv}) = 0;
        end

        %% ---------- parameter setting ----------
        MM = [nCluster 2*nCluster 3*nCluster 4*nCluster 5*nCluster];
        DDD= [nCluster 2*nCluster 3*nCluster 4*nCluster 5*nCluster];
       
        BT = [0.0001 0.001 0.01 0.1 1 10 100 1000];
        CP = [0.0001 0.001 0.01 0.1 1 10 100 1000];
        DT = [0.0001 0.001 0.01 0.1 1 10 100 1000];
       
      
        %% ================= grid search =================
        for rp00 = 1:length(DDD)
             param.ddd= DDD(rp00);
        for rp0 = 1:length(MM)
            param.m = MM(rp0);
            for rp2 = 1:length(BT)
                for rp3 = 1:length(CP)
                    for rp4 = 1:length(DT)
                        
                                                                                                                                             
                        param.gamma   = BT(rp2);
                        param.alpha  = CP(rp3);
                        param.beta = DT(rp4);
                       
                        tic;
                        [F,X_complete,iter_num] = DTBG(X, TIND, ind_0, n, param); 
                        F = F ./ (repmat(sqrt(sum(F .^ 2, 2)), 1,  param.c)+eps);
                        time = toc;
                       
                        res_runs = zeros(num_cluster_runs, 4);
                        for rep = 1:num_cluster_runs
                                 predY = kmeans(F, param.c, ...
                                'maxiter', 1000, ...
                                'replicates', 20, ...
                                'emptyaction', 'singleton');

                            metrics = Clustering8Measure(Y, predY);
                            res_runs(rep,:) = reshape(metrics(1:4), 1, 4);
                        end
                        % Average all 20 runs without selection by ground-truth ACC.
                        res = mean(res_runs, 1);
                        res_std = std(res_runs, 0, 1); % Sample standard deviation (N-1).
                        ACC = res(1);
                        metric_names = {'ACC','NMI','Purity','Fscore'};
                        for metric_idx = 1:numel(metric_names)
                            fprintf('%s = %.6f +/- %.6f\n', metric_names{metric_idx}, ...
                                res(metric_idx), res_std(metric_idx));
                        end

                        %% ---------- update best ----------
                        if ACC > best_ACC
                            best_ACC   = ACC;
                            best_res   = res;
                            best_std   = res_std;
                            best_param = param;
                            best_iter  = iter_num;
                            best_time  = time;
                        end

                        %% ---------- save all results ----------
                        row_table = table( ...
                                mr, ...
                                param.m, param.ddd, param.gamma, param.alpha, param.beta, ...
                                res(1), res(2), res(3), res(4), ...
                                res_std(1), res_std(2), res_std(3), res_std(4), ...
                                iter_num, time, ...
                                'VariableNames', varNames);

                        writetable(row_table, excel_name, 'WriteMode', 'append');
                       fprintf(['[mr=%.1f | m=%d | d=%d | gamma=%.4f | alpha=%.4f | beta=%.4f | ' ...
                        'iter=%d | time=%.2fs | ACC=%.4f +/- %.4f | Best_ACC=%.4f | ' ...
                        'c=%d | num_view=%d | n=%d]\n'], ...
                         mr, param.m, param.ddd, param.gamma, param.alpha, param.beta, ...
                         iter_num, time, ACC, res_std(1), best_ACC, ...
                         param.c, num_view, n);
                        % 
                        % fprintf('[mr=%.1f | m=%d | gamma=%.2f] ACC=%.4f | Best=%.4f\n', ...
                        %     mr, param.m, param.gamma, ACC, best_ACC);

                        
                    end
                end
            end
        end
        end

        %% ================= 保存当前 miss_rate 的 Best_ACC =================
       best_table = table( ...
             mr, ...
             best_param.m, best_param.ddd, best_param.gamma, best_param.alpha, best_param.beta, ...
             best_ACC, best_res(2), best_res(3), best_res(4), ...
             best_std(1), best_std(2), best_std(3), best_std(4), ...
             best_iter, best_time, ...
             'VariableNames', varNames);

        writetable(best_table, excel_name, ...
            'Sheet', 'Best_ACC', ...
            'WriteMode', 'append');

        fprintf('>>> BEST (mr=%.1f): ACC=%.4f +/- %.4f\n\n', mr, best_ACC, best_std(1));

        clear X Y
    end
end

function [UU,X,iter_num] = DTBG(X,TIND, ind_0,n, param)
num_view = length(X);

gamma = param.gamma;
alpha =  param.alpha;
beta = param.beta;

m=param.m;
c= param.c;
ddd = param.ddd;
miu1 =100; 
miu2 = 100; 

I = eye(n); 
missingindex = cell(1,num_view);
for iv = 1:num_view
    temp_index = ones(1,n);
    temp_index(1,ind_0{iv}) = 0;
    missingindex{iv} = temp_index;
end

if size(X{1},2)~=n
    for iv = 1:num_view
        X{iv} = X{iv}';
    end
end
for iv = 1:num_view
    X{iv} = NormalizeFea(X{iv},0);
end
clear iv

dims = zeros(1,num_view);
for iv = 1:num_view
    dims(iv) = size(X{iv},1);
end
%% initialize
A = zeros(ddd,m);

SS = zeros(m,n);
% SS = SS ./ sum(SS,1);
for iv = 1:num_view
   di = size(X{iv},1); 
   P{iv} = zeros(di,ddd);
   C{iv} = SS;
   D{iv} = SS;
   M{iv} = zeros(m, n);   
   Y2{iv} = zeros(m, n);
end

for iv = 1:num_view
     Z{iv} = zeros(m, n);
     Y1{iv} = zeros(m, n);
end
rou= 1.5;
max_miu = 10e10;
sX = [m, n, num_view];
%% Training
MAX_ITER =1000;
for iter = 1:MAX_ITER 
  
     %% 1.更新P{v}
    for iv = 1:num_view
        E{iv} = X{iv}.*(repmat(missingindex{iv},dims(iv),1))*(C{iv}+D{iv})'*A';
        [U_b,~,V_b] = svd(E{iv},"econ");
        P{iv} = U_b*V_b';
    end
     %% 2.更新A

    TempB = 0;
    for iv = 1:num_view
        TempB = TempB + (P{iv})'*(X{iv}.*(repmat(missingindex{iv},dims(iv),1)))*(C{iv}+D{iv})';
    end
    [U,~,V] = svd(TempB,"econ");
    A = U*V'; % A 融合所有视图
     %% 3.更新C（投影梯度；保留原重构项简化）
     for iv = 1:num_view
          KK =2*A'*P{iv}'* (X{iv}.*(repmat(missingindex{iv},dims(iv),1)))-2*D{iv}.*(repmat(missingindex{iv},m,1))+miu1*Z{iv}-Y1{iv};
          H = 2*gamma*(D{iv}*D{iv}');
          h = 2*missingindex{iv}+miu1;
          C{iv} = pgUpdateOriginalProjection(C{iv},H,h,KK);
     end
     %% 4.更新D（使用更新后的C）
     for iv = 1:num_view
          KK2 =2*A'*P{iv}'* (X{iv}.*(repmat(missingindex{iv},dims(iv),1)))-2*C{iv}.*(repmat(missingindex{iv},m,1))+miu2*M{iv}-Y2{iv};
          H = 2*gamma*(C{iv}*C{iv}');
          h = 2*missingindex{iv}+miu2;
          D{iv} = pgUpdateOriginalProjection(D{iv},H,h,KK2);
     end
   %% 5.更新张量Z SP范数

        C_tensor = cat(3, C{:, :});
        Y1_tensor = cat(3, Y1{:, :});
        bvec_C = C_tensor(:);
        bvec_Y1 = Y1_tensor(:);
        [bvec_Z, ~] = wshrinkObj( bvec_C + (1 / miu1) *  bvec_Y1, alpha / miu1, sX, 0, 1);
        Z_tensor = reshape(bvec_Z, sX);
        
     %% 6.更新张量M 21范数
    M_tensor = cat(3, M{:,:});
    D_tensor = cat(3, D{:,:});
    Y2_tensor = cat(3, Y2{:,:});   
    for v = 1:num_view
        M_tensor(:,:,v) = prox_l21(D_tensor(:,:,v)+(1/miu2)*Y2_tensor(:,:,v),beta/miu2); 
        M{v} = M_tensor(:,:,v);
    end
    
  
     %% 更新 Y 
        % Z_tensor = permute(Z_tensor, [1 3 2]); 
        Y1_tensor = C_tensor-Z_tensor;
        Y2_tensor = D_tensor - M_tensor;
        for v = 1:num_view
            Y1{v} = Y1{v} + miu1*Y1_tensor(:,:,v);       
        end
        for v = 1:num_view
           Y2{v} = Y2{v} + miu2*Y2_tensor(:,:,v);     
        end
        %% 更新 miu
        miu1 = min(rou * miu1, max_miu);
        miu2 = min(rou * miu2, max_miu);
    %% 7.记录obj
    if iter == 1
        Z_tensor_pre = zeros(size(Z_tensor));
        M_tensor_pre = zeros(size(M_tensor));
    end
   
    % UU = UU(:,1:c);
   
%% ===== 收敛判断  =====
if iter == 1
    for v = 1:num_view
        C_pre{v} = C{v};
        D_pre{v} = D{v};
    end
end
% ===== C / D 收敛 =====
    curr_diff_C = 0;
    curr_diff_D = 0;

    if iter > 1
    for v = 1:num_view
        curr_diff_C = curr_diff_C + norm(C{v} - C_pre{v}, 'fro') / (norm(C_pre{v}, 'fro') + eps);
        curr_diff_D = curr_diff_D + norm(D{v} - D_pre{v}, 'fro') / (norm(D_pre{v}, 'fro') + eps);
    end

    curr_diff_C = curr_diff_C / num_view;
    curr_diff_D = curr_diff_D / num_view;
    curr_diff_Z = norm(Z_tensor - Z_tensor_pre, 'fro') / (norm(Z_tensor_pre, 'fro') + eps);
    curr_diff_M = norm(M_tensor - M_tensor_pre, 'fro') / (norm(M_tensor_pre, 'fro') + eps); 
       % ===== 总收敛 =====
        actual_max_change = max([ ...
        curr_diff_Z, ...
        curr_diff_M, ...
        curr_diff_C, ...
        curr_diff_D ...
        ]);
        
        if actual_max_change < 1e-6
            fprintf('Algorithm convergence reached at iteration %d\n', iter);
            break;
        end
    end

    % --- 更新历史记录 (务必保留) ---
    Z_tensor_pre = Z_tensor;
    M_tensor_pre = M_tensor;
    for v = 1:num_view
        C_pre{v} = C{v};
        D_pre{v} = D{v};
    end
end
 S=0;
    for v = 1:num_view
        S = S+ C{v};
    end
    [UU,~,V]=svd(S','econ');
     UU = UU(:,1:c);
iter_num=iter;
fprintf("iter=%d\n", iter);

end



function U = pgUpdateOriginalProjection(U,H,h,rhs)
% Correct coupling gradient H*U; retain original reconstruction simplification.
% Fixed step uses a Lipschitz upper bound for the simplified quadratic.
L = max(sum(abs(H),2)) + max(h);
for k = 1:200
    G = H*U + U.*repmat(h,size(U,1),1) - rhs;
    next = SimplexProj((U-G/L)');
    next = next';
    relativeStep = norm(next-U,'fro')/max(1,norm(U,'fro'));
    U = next;
    if relativeStep <= 1e-7
        break;
    end
end
end
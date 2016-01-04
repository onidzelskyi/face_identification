function [stat] = calcDistance(X, test, type)
% Calc chi squared statistic

% type = 1 Chi-square distance C = sum((X-test).^2/test)
% type = 2 Chi-square distance C = sum((X-test).^2/(X+test))
% type = 3 Chi-square distance C = sum((X-test).^2/(X+test)) for non-zero
% values
% type = 4 Euclidian distance

if type==1
    stat = sum(bsxfun(@rdivide,bsxfun(@minus, X, test).^2, test),2);
elseif type==2
    stat = sum(bsxfun(@rdivide,bsxfun(@minus, X, test).^2, bsxfun(@plus, X, test)),2);
elseif type==3
    fun = @(block) dist_chi2(block.data,test);
    stat = blockproc(X, [1,size(X,2)], fun);
elseif type==4
    stat = pdist2(X,test);
else
    fprintf('Type %d not supported\n', type);
end

end
function D = dist_chi2(A,B)

d=size(A,1);
d1=size(B,1);
if (d ~= d1)
    error('column length of A (%d) != column length of B (%d)\n',d,d1);
end

nom=(A-B).^2;
den=(A+B);
rlt=nom(den~=0)./den(den~=0);

D=sum(rlt);
end
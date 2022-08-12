%%%% a5 normalized on mean before injury(4 times)
%%%% a6 mean normalized for all  mice

a=b1;
a3=a(1:4,:)
a4=nanmean(a3)
a5=[];
a7=size(a)
for i=1:a7(2)
a5(:,i)= a(:,i)/a4(1,i);
end
a6=nanmean(a5')

before = a5(1:4, :);
before=before(:);
after1=a5(5, :);
after1=after1';
after = a5(5:end, :);
after=after(:);

figure
plot(a5,'.','MarkerSize', 20)
ylabel('Number of neurons')
xlabel('Session')
hold on
plot(a6, 'k', 'LineWidth', 2)
hold off

nanmean(before)
nanstd(before)/length(before)
nanmean(after1)
nanstd(after1)/length(after1)
nanmean(after)
nanstd(after)/length(after)
[nanmean(before) nanstd(before)/length(before) nanmean(after1) nanstd(after1)/length(after1) nanmean(after) nanstd(after)/length(after)]



[R,P] = ttest2(before, after1)
[R,P] = ttest2(before, after)
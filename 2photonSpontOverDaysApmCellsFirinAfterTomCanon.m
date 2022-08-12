x = h5read('02_b3_000_000.hdf5','/estimates/F_dff');
x=xx;
b=size(x)
%x1=posTrials02
peak_num = [];
Prom = [];

peak_num = [];
Prom = [];
for l=1:b(2)
w=x(:,l);
figure
plot (w)
end
%     ampD = [ampD Damp];
%     Dcoord = [Dcoord DX]
%     hold off
% end
%Damp=mean(w(DX-shag/2:DX+shag/2))


x1=[1:b(2)];
 x2 = [];%%% trails to exclude
 x1=x1(~ismember(x1,x2));




for l=1:length(x1)%b(2)
    w=x(:,x1(l));
    
    
    figure
    plot (w)
    s_y = 4*std(w);
    smoothy = sgolayfilt(w,7,21);
    sm_y=4*std(smoothy);
    
    figure;
            subplot(211);
            findpeaks(w,'MinPeakProminence',s_y)
            [filty,filtx,ww,p] = findpeaks(w,'MinPeakProminence',s_y);
            unfilt{1,l} = [filtx',filty'];
            unfiltP{1,l} = [p'];
            title('Unfiltered')
            
            subplot(212);
            findpeaks(smoothy,'MinPeakProminence',sm_y)
            [filty,filtx,ww,p] = findpeaks(smoothy,'MinPeakProminence',sm_y);
            filt{1,l} = [filtx',filty'];
            filtP{1,l} = [p'];
            pn = size(filtx);
            peak_num = [peak_num pn(1)];
            Prom = [Prom mean(p)];
            
            title('Filtered')
%     hold on
%     Damp=mean(w(DX-shag/2:DX+shag/2))
%     
%     if isfinite(Damp)==1
%         plot([DX], [Damp], '^r', 'MarkerFaceColor','r')
%     end    
% %plot([w(ipt), ipt], ylim, '--')
    
    
    end
%     ampD = [ampD Damp];
%     Dcoord = [Dcoord DX]
%     hold off
% end
%Damp=mean(w(DX-shag/2:DX+shag/2))
result = [mean(peak_num) length(x1) mean(Prom) b(1)]

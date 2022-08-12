xx = h5read('02_b3_000_000.hdf5','/estimates/F_dff');
load('02_b7_000_001.mat');



b=size(xx)
%x1=posTrials02
peak_num = [];
Prom = [];

SorWh=size(fields(info))
if SorWh(1)==24
    x=info.frame;
    x=x+1;
else
   i=1 
end 
i=1

for l=1:b(2)
    w=[];
    w1=[];
    w2=[];
    i=1
    w=xx(:,l);
    
    
    for n = 1:(length(x)-1)
        w1{i} = w(x(n):(x(n)+76),:);
        i = i+1 
    end
    
    newtry = w1{1};
    cc=size(w1,2);
    for nn=2:cc-2;
        newtry = newtry+w1{nn};
        w2= newtry/cc;
     
    end
    rr = [0:76]';
    for nnn=(1:length(w2(1,:))); 
        figure(nnn)
        plot(rr, w2(:,nnn))
        title(['cell : ',num2str(l)])
        
        s_y = std(w2);
%         %findpeaks(w2,'MinPeakProminence',s_y)
        [filty,filtx,ww,p] = findpeaks(w2,'MinPeakProminence',s_y);
        Prom = [Prom max(p)];
        pause
    end
    
%figure
%plot (w)
end












 x1=[1:b(2)];
 x2 = [];%%% trails to exclude
 x1=x1(~ismember(x1,x2));
 
 
 w=[];
 w1=[];
 w2=[];
 newtry =[];
 cc=[];
 i=1;
 l=1;
 n=1;
 nn=1;
 nnn=1;
 WH=[];
 Prom=[];
for l=1:length(x1)%b(2)
    w3=[];
    w4=[];
    w5=[];
    i=1 
    w3=xx(:,x1(l));
    
    
    for n = 1:(length(x)-1)
        w4{i} = w3(x(n):(x(n)+76),:);
        i = i+1 
    end
    
    newtry = w4{1};
    cc=size(w4,2);
    for nn=2:cc-2;
        newtry = newtry+w4{nn};
        w5= newtry/cc;
    end    
    
    for nnn=(1:length(w5(1,:))); 
        %figure(nnn)
        
        %title(['cell : ',num2str(l)])
    
        %figure(nnn)
        %plot(rr, w2(:,nnn))
        %title(['cell : ',num2str(l)])
    
        s_y = std(w5);
        %smoothy = sgolayfilt(w2,7,21);
        %sm_y=4*std(smoothy);
        
        %figure;
        %subplot(211);
        %findpeaks(w2,'MinPeakProminence',s_y) 107,109,110,111
        [filty,filtx,ww,p] = findpeaks(w5,'MinPeakProminence',s_y);
        %unfilt{1,l} = [filtx',filty'];
        %unfiltP{1,l} = [p'];
        %title('Unfiltered')
            
        %subplot(212);
        %findpeaks(smoothy,'MinPeakProminence',sm_y)
        %[filty,filtx,ww,p] = findpeaks(smoothy,'MinPeakProminence',sm_y);
        %filt{1,l} = [filtx',filty'];
        %filtP{1,l} = [p'];
        pn = size(filtx);
        peak_num = [peak_num pn(1)];
        Prom = [Prom max(p)];
        WH = [WH max(ww)]
            
    end
        
        
        
 end
%     rr = [0:76]';
%     for nnn=(1:length(w2(1,:))); 
%         figure(nnn)
%         plot(rr, w2(:,nnn))
%         title(['cell : ',num2str(l)])
%     pause
%     end
%     
    
    
    
    
    
    
    
    
    
%     hold on
%     Damp=mean(w(DX-shag/2:DX+shag/2))
%     
%     if isfinite(Damp)==1
%         plot([DX], [Damp], '^r', 'MarkerFaceColor','r')
%     end    
% %plot([w(ipt), ipt], ylim, '--')
    
    
   % end
%     ampD = [ampD Damp];
%     Dcoord = [Dcoord DX]
%     hold off
% end
%Damp=mean(w(DX-shag/2:DX+shag/2))

ma=flip((sort(Prom)));  
maWH=flip((sort(WH)));
    result=ma(1:10);  
    resultWH = maWH(1:10);%top ten
    max10=mean(result);
    maxWH10 = mean(resultWH);
 resultat = [ length(x1) mean(Prom) NaN max(Prom) mean(WH) NaN max(WH)]  
 resultat = [ length(x1) mean(Prom) max10 max(Prom) mean(WH) maxWH10 max(WH)]

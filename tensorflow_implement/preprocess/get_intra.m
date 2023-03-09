
load data/iemocap/imse1.mat 
a=length(imse1);               
A1=imse1{1,1};                     
parfor j=2:a
    fprintf('xie=%d\n',j);
    A1=[A1;imse1{j,1}];                
end

load dataset/iemocap/context_label1.txt   
load dataset/iemocap/numSeg_1.mat   
y1=[];                                 
numSeg_1=numSeg_1';
context_label=context_label1;
for j=1:a
    fprintf('j=%d\n',j);
    l=numSeg_1(j,1);                    
    value=context_label(j,:);              
    parfor k=1:l
        mid(k,:)=value; 
    end
    y1=[y1;mid];       
    clear mid;
end
save data/iemocap/yyy1.mat y1 -v7.3 
save data/iemocap/session11.mat A1 y1 -v7.3  
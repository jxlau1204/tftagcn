wavpath = 'data/iemocap/session1/';
context='data/iemocap/Wav_context_1.txt';
cd(wavpath);                       
filelist = dir('*.wav');   
filelist = struct2cell(filelist); 
filelist = filelist(1,:)';
filename = cell(length(filelist),1);
contextname=cell(length(context),1);
fid=fopen(context,'r');
INDEX=1;
while ~feof(fid)
   str = fgetl(fid);  
   contextname{INDEX,1}=str;
   INDEX=INDEX+1;
end

w=256; 
n=256; 
ov=128;

for i=1:length(contextname)
    for j=1:length(filelist)
        a = filelist(j);    
        [pathstr,name,ext] = fileparts(a{1});
        if(contextname{i,1}==name)
            filename{i,1} = name;
        end
    end
end
clear filelist a i

segmentLength = 0.265; 
numSeg_1=[]; 
for i=1:length(filename)
    wavfile = [wavpath,filename{i},ext];
    [x1,fs1] = audioread(wavfile);
    fs=16000;
    x=resample(x1,fs,fs1);
    x=x(:,1);

    d=segmentLength*16000;   
    move=0.025*16000;        
    x_start = 1;
    k=1; 
    while 1
        x_end = x_start + d-1;
        if x_end > length(x(:,1))
            break;
        end
        t = x(x_start:x_end,:);  

        yy(k,:,:) = t;
        x_start = x_start + move; 
        k=k+1;
    end

    kk=length(yy(:,1));
    numSeg_1=[numSeg_1,kk];  
    for L=1:kk
        xx=double(yy(L,:)');  
        [S,~,~,~]=spectrogram(xx,w,ov,n,fs);
        S=log(1+abs(S));
        S(1,:) = [];
        z(L,:,:)=S';
    end
    imse1{i,:}=z;
    clear x_end L kk yy xx z;
end
disp(size(imse1))
 save('data/iemocap/imse1','imse1','-v7.3')
 save('data/iemocap/numSeg_1','numSeg_1','-v7.3')
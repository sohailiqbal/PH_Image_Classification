function [CCmatrix] = solidities_kmeans(image,NoC,d)
%SOLIDITIES_KMEANS Summary of this function goes here
%   Detailed explanation goes here
       
imshow(image);

image=rgb2gray(image);

%Im = averagefilter(image, [20 20]);
%figure, imshow(C5), figure, imshow(J)


%----these two commands use relnoise-------------
ImgN=relnoise(image,3,0.3,'disk');
%ImgN=relnoise(image,3,0.3,'disk');

Im = uint8(255 * mat2gray(ImgN));
%------------------------------------------------

%Im=medfilt2(Im);

ImMean=imsegkmeans(Im,NoC);
Matrix=Im;
[a b]=size(Matrix);

X=zeros(a, b, NoC);
CC=cell(NoC,1);

for k=1:NoC
M=Matrix;
[n m]=size(M);
for i=1:n
    for j=1:m
        if (ImMean(i,j)~= k )         
            M(i,j)=0;
        end
    end
end


X(:,:,k)=M;  %output of required image in greyscale


BW2 = edge(M,'canny'); % edge detection
s  = regionprops(BW2, 'centroid','solidity');


centroids = cat(1, s.Centroid); % centroids
sol=cat(1,s.Solidity); % solidities

cent=Ccircs(centroids,sol,d); 


A=((500/NoC)*(k-1)).*ones(size(cent,1),1);
V=[cent A];
CC{k,1}=V;


%-------------------------------------------------------------------
%to see image and the centroids on it. Also uncomment imshow(image) above
%------------------------------------------------------------------
hold on
%plot(cent(:,1), cent(:,2),'linestyle','none','marker','o','MarkerSize',10,'MarkerFaceColor','red')
plot(cent(:,1), cent(:,2), 'o')
hold on


end


CCmatrix=cell2mat(CC);

end




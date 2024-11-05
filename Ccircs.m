function [out]=Ccircs(cen,circle, d)
m=length(circle);

s=1;
for i=1:m
   
    %you can change the following parameter but for Covid-19 images
    % 0.8 works file.
   if circle(s)<d
       circle(s)=[];
       cen(s,:)=[];
   else 
           s=s+1;
  
   end
  
end
out=cen;
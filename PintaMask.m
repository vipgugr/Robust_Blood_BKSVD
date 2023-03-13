function [] = PintaMask(I,mask2d)
band=2 %R=1, G=2, B=3
overlaped=I;
for i=1:3
    channel=overlaped(:,:,i);
    if i==band
        channel(mask2d)=255;
    else
        channel(mask2d)=0;
    end
    overlaped(:,:,i)=channel;
end
%figure()
imshow(overlaped)

end
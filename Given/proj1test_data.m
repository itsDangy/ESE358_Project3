% Program to check test  data for Project 3
% Prof. M. Subbarao, SBU, ECE, 
%Sample output of Canny's edge detector and Harris corner detector are
%shown.

p1g3 = imread('pic1grey300.jpg');
figure, imshow(p1g3) , title('pic1grey300')

%p1g3ep = edge(p1g3,'prewitt');
%figure, imshow(p1g3ep) , title('prewitt edges pic1grey300')

p1g3ec = edge(p1g3,'canny');
figure, imshow(p1g3ec) ,  title('Canny edges pic1grey300')
imwrite(p1g3ec,'pic1greyCannyEdges.jpg');

corners = detectHarrisFeatures(p1g3, 'MinQuality' , 0.0001 , 'FilterSize' , 7);
figure, imshow(p1g3); title('Harris Corners, pic1grey300'), hold on;
plot(corners.selectStrongest(200));
  
p2g3 = imread('pic2grey300.jpg');
figure, imshow(p2g3) , title('pic2grey300')

%p2g3ep = edge(p2g3,'prewitt');
%figure, imshow(p2g3ep) , title('prewitt edges pic1grey300')

p2g3ec = edge(p2g3,'canny');
figure, imshow(p2g3ec) ,  title('Canny edges pic2grey300')
imwrite(p2g3ec,'pic2greyCannyEdges.jpg');

corners = detectHarrisFeatures(p2g3, 'MinQuality' , 0.0001 , 'FilterSize' , 7);
figure, imshow(p2g3); title('Harris Corners, pic2grey300'), hold on;
plot(corners.selectStrongest(200));
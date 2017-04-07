## K-Means Image Segmentation
Uses K-Means to segment an image into k colors.

Optionally uses a rescale factor to speed up mean computation.
Rescaling to 25% image size, for example, leads to very similar end color means, while taking ~1/16th the time.
Image is reconstructed using original resolution.

Original image:

![alt tag](http://i.imgur.com/GXF4SlB.jpg)

k=8 100% scale, k=8 25% scale. 10 computation iterations.

![alt tag](http://i.imgur.com/xaLJSj2.png)
![alt tag](http://i.imgur.com/fowj6hO.png)

k=3 25% scale. 10 computation iterations.

![alt tag](http://i.imgur.com/xJH7mjY.png)

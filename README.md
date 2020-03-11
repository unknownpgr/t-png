# Implementing an image that changes when clicked
## Concept
Some websites have a different background color when viewed on the main screen and when the image is clicked on a large screen. In this case, the png image can set the transparency of pixels so that the image can look different according to the background color.
I developed an algorithm that automatically generates these images.
In particular, the existing algorithm can be implemented only for black and white images, but the algorithm I developed can also implement color images by applying vector operations.
## Algorithm
Refer to [tpng.pdf](./tpng.pdf)(Written in Korean)
## Result
(I developed two alogrithms but there are no significant differences.)
![](./sol_1.png)  
The image above looks different depending on whether the background color is black or white.
## Source code
See [gen.py](./gen.py)
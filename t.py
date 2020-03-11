import numpy as np
import cv2


def ps1(x, y):
    """
    Get nearest solution of |ax+y|=0 for a
    x,y = array-like objeect
    """
    return -np.dot(x, y)/np.dot(x, x)


def getAlpha(x, y, b1, b2):
    """
    x,y,b1,b2 = numpy vector
    """
    return ps1(b1-b2, x-y-b1+b2)


def getSolution(x, y, b1, b2):
    alpha = getAlpha(x, y, b1, b2)
    value = (x+y-b1-b2)/(2*alpha)+(b1+b2)/2
    return np.concatenate((value, [alpha*255]))


imA = 'a.jpg'
imB = 'b.jpg'
D = 0.2

imA = cv2.imread(imA)
imB = cv2.imread(imB)

H = max(imA.shape[0], imB.shape[0])
W = max(imA.shape[1], imB.shape[1])

# Add padding to image
ia = np.ones([H, W, 3])*255
ia[:imA.shape[0], :imA.shape[1]] = np.array(
    imA, np.float32)[:, :, :3]*(1-D)+255*D

# Add padding to image
ib = np.zeros([H, W, 3])
ib[:imB.shape[0], :imB.shape[1]] = np.array(imB)[:, :, :3]*(1-D)

b1 = np.ones([H, W, 3])*255
b2 = np.zeros([H, W, 3])
c = np.zeros([H, W, 4])

for i in range(H):
    print(f'{i} / {H} ({round(i*100/H,1)}%)')
    for j in range(W):
        c[i, j] = getSolution(ia[i, j], ib[i, j], b1[i, j], b2[i, j])
c[c > 255] = 255
c[c < 0] = 0
x = c.astype(np.uint8)
cv2.imwrite('r.png', x)
cv2.waitKey(0)

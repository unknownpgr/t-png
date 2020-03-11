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


def func1(x, y, b1, b2):
    alpha = getAlpha(x, y, b1, b2)
    value = (x+y-b1-b2)/(2*alpha)+(b1+b2)/2
    return np.concatenate((value, [alpha*255]))


def func2(x, y, bx, by):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    y1 = y[0]
    y2 = y[1]
    y3 = y[2]

    bx1 = bx[0]
    bx2 = bx[1]
    bx3 = bx[2]

    by1 = by[0]
    by2 = by[1]
    by3 = by[2]

    r = (bx2**2*x1 + bx3**2*x1 + 2*by1**2*x1 + by2**2*x1 + by3**2*x1 + 2*bx1**2*y1 + bx2**2*y1 + bx3**2*y1 + by2**2*y1 + by3**2*y1 - bx1*bx2*x2 - bx1*bx3*x3 - 2*bx1*by1*x1 + bx1*by2*x2 - bx2*by1*x2 - 2*bx2*by2*x1 + bx1*by3*x3 - bx3*by1*x3 - 2*bx3*by3*x1 + by1*by2*x2 + by1*by3*x3 + bx1*bx2*y2 + bx1*bx3*y3 - 2*bx1*by1 *
         y1 - bx1*by2*y2 + bx2*by1*y2 - 2*bx2*by2*y1 - bx1*by3*y3 + bx3*by1*y3 - 2*bx3*by3*y1 - by1*by2*y2 - by1*by3*y3)/(2*(by1*x1 - 2*bx2*by2 - 2*bx3*by3 - bx1*x1 - bx2*x2 - bx3*x3 - 2*bx1*by1 + by2*x2 + by3*x3 + bx1*y1 + bx2*y2 + bx3*y3 - by1*y1 - by2*y2 - by3*y3 + bx1**2 + bx2**2 + bx3**2 + by1**2 + by2**2 + by3**2))
    g = (bx1**2*x2 + bx3**2*x2 + by1**2*x2 + 2*by2**2*x2 + by3**2*x2 + bx1**2*y2 + 2*bx2**2*y2 + bx3**2*y2 + by1**2*y2 + by3**2*y2 - bx1*bx2*x1 - bx2*bx3*x3 - 2*bx1*by1*x2 - bx1*by2*x1 + bx2*by1*x1 - 2*bx2*by2*x2 + bx2*by3*x3 - bx3*by2*x3 - 2*bx3*by3*x2 + by1*by2*x1 + by2*by3*x3 + bx1*bx2*y1 + bx2*bx3*y3 - 2*bx1*by1 *
         y2 + bx1*by2*y1 - bx2*by1*y1 - 2*bx2*by2*y2 - bx2*by3*y3 + bx3*by2*y3 - 2*bx3*by3*y2 - by1*by2*y1 - by2*by3*y3)/(2*(by1*x1 - 2*bx2*by2 - 2*bx3*by3 - bx1*x1 - bx2*x2 - bx3*x3 - 2*bx1*by1 + by2*x2 + by3*x3 + bx1*y1 + bx2*y2 + bx3*y3 - by1*y1 - by2*y2 - by3*y3 + bx1**2 + bx2**2 + bx3**2 + by1**2 + by2**2 + by3**2))
    b = (bx1**2*x3 + bx2**2*x3 + by1**2*x3 + by2**2*x3 + 2*by3**2*x3 + bx1**2*y3 + bx2**2*y3 + 2*bx3**2*y3 + by1**2*y3 + by2**2*y3 - bx1*bx3*x1 - bx2*bx3*x2 - 2*bx1*by1*x3 - bx1*by3*x1 + bx3*by1*x1 - 2*bx2*by2*x3 - bx2*by3*x2 + bx3*by2*x2 - 2*bx3*by3*x3 + by1*by3*x1 + by2*by3*x2 + bx1*bx3*y1 + bx2*bx3*y2 - 2*bx1*by1 *
         y3 + bx1*by3*y1 - bx3*by1*y1 - 2*bx2*by2*y3 + bx2*by3*y2 - bx3*by2*y2 - 2*bx3*by3*y3 - by1*by3*y1 - by2*by3*y2)/(2*(by1*x1 - 2*bx2*by2 - 2*bx3*by3 - bx1*x1 - bx2*x2 - bx3*x3 - 2*bx1*by1 + by2*x2 + by3*x3 + bx1*y1 + bx2*y2 + bx3*y3 - by1*y1 - by2*y2 - by3*y3 + bx1**2 + bx2**2 + bx3**2 + by1**2 + by2**2 + by3**2))
    a = (by1*x1 - 2*bx2*by2 - 2*bx3*by3 - bx1*x1 - bx2*x2 - bx3*x3 - 2*bx1*by1 + by2*x2 + by3*x3 + bx1*y1 + bx2*y2 + bx3*y3 - by1*y1 - by2*y2 - by3 *
         y3 + bx1**2 + bx2**2 + bx3**2 + by1**2 + by2**2 + by3**2)/(bx1**2 - 2*bx1*by1 + bx2**2 - 2*bx2*by2 + bx3**2 - 2*bx3*by3 + by1**2 + by2**2 + by3**2)
    return np.array([r, g, b, a*255])


def func3(x, y, _, __):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    rgba = [(130050*x1 - 65025*x2 - 65025*x3 + 260100*y1 + 65025*y2 + 65025*y3) /
            (2*(255*y1 - 255*x2 - 255*x3 - 255*x1 + 255*y2 + 255*y3 + 195075)),
            (130050*x2 - 65025*x1 - 65025*x3 + 65025*y1 + 260100*y2 + 65025*y3) /
            (2*(255*y1 - 255*x2 - 255*x3 - 255*x1 + 255*y2 + 255*y3 + 195075)),
            (130050*x3 - 65025*x2 - 65025*x1 + 65025*y1 + 65025*y2 + 260100*y3) /
            (2*(255*y1 - 255*x2 - 255*x3 - 255*x1 + 255*y2 + 255*y3 + 195075)),
            y1/3 - x2/3 - x3/3 - x1/3 + y2/3 + y3/3 + 255]
    return np.array(rgba)


pathX = 'a.jpg'
pathB = 'b.jpg'
D = 0.2
scale = 0.5

orgX = cv2.imread(pathX)
orgY = cv2.imread(pathB)

H = max(orgX.shape[0], orgY.shape[0])
W = max(orgX.shape[1], orgY.shape[1])

# Add padding to image and color shift
imgX = np.ones([H, W, 3])*255
imgX[: orgX.shape[0], : orgX.shape[1]] = np.array(
    orgX, np.float32)[:, :, :3]*(1-D)+255*D
imgY = np.zeros([H, W, 3])
imgY[:orgY.shape[0], :orgY.shape[1]] = np.array(orgY)[:, :, :3]*(1-D)

# Scaling
imgX = cv2.resize(imgX, (int(W*scale), int(H*scale)))
imgY = cv2.resize(imgY, (int(W*scale), int(H*scale)))
W = int(W*scale)
H = int(H*scale)

bgrX = np.ones([H, W, 3])*255
bgrY = np.zeros([H, W, 3])


def solve(x, y, b1, b2, func):
    result = np.zeros([H, W, 4])
    for i in range(H):
        if i % 50 == 0:
            print(f'{i} / {H} ({round(i*100/H,1)}%)')
        for j in range(W):
            result[i, j] = func(x[i, j], y[i, j], b1[i, j], b2[i, j])
    result[result > 255] = 255
    result[result < 0] = 0
    return result.astype(np.uint8)


r = solve(imgX, imgY, bgrX, bgrY, func1)
cv2.imwrite('sol_1.png', r)

r = solve(imgX, imgY, bgrX, bgrY, func3)
cv2.imwrite('sol_2.png', r)

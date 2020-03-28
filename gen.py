import numpy as np
import cv2


def ps1(x, y):
    """
    Get optimal solution of |ax+y|=0 for a
    (x,y = array-like objeect)
    """
    return -1*np.dot(x, y)/np.dot(x, x)


def _func1_get_alpha(x, y, b1, b2):
    """
    Get alpha value for algorithm 1
    x,y,b1,b2 = numpy vector
    """
    return ps1(b1-b2, x-y-b1+b2)


def func1(x, y, b1, b2):
    """
    Make a pixel to be x on background b1 and y on background b2.
    Optimize alpha first and then calculate rgb from a.
    """
    alpha = _func1_get_alpha(x, y, b1, b2)
    rgb = (x+y-b1-b2)/(2*alpha)+(b1+b2)/2
    return np.concatenate((rgb, [alpha*255]))


def func2(x, y, bx, by):
    """
    Make a pixel to be x on background bx and y on background by.
    Optimize alpha and rgb at once.
    """
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

    # I used matlab to calculate this formula.
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
    """
    Make a pixel to be x on background white and y on background black.
    3rd and 4th parameter is nothing.
    func3 is equal to func2(x,y,[255,255,255],[0,0,0]).
    """

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


# Path of input image
pathX = 'a.jpg'
pathY = 'b.jpg'

# Color shift
D = 0.2

# Size scale factor
scale = 0.5

orgX = cv2.imread(pathX)
orgY = cv2.imread(pathY)

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

# Background image
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
    result = result.astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'unknownpgr@gmail.com', (10, 20),
                font, 0.5, (128, 128, 128), 1, cv2.LINE_AA)

    return result


r = solve(imgX, imgY, bgrX, bgrY, func1)
cv2.imwrite('sol_1.png', r)

r = solve(imgX, imgY, bgrX, bgrY, func3)
cv2.imwrite('sol_2.png', r)

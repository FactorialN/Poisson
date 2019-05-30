import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pylab as pyl
import matplotlib
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

class Mywindow(QWidget):

    def __init__(self, p1, p2, n, m):
        super(Mywindow, self).__init__()
        self.setMouseTracking(True)
        self.p1 = p1
        self.p2 = p2
        self.n = n
        self.m = m
        xx = p1.shape[1] + p2.shape[1] + 50
        self.xx = xx
        yy = max(p2.shape[0], p2.shape[0]) + m + 50
        self.setFixedSize(xx, yy)
        self.setWindowTitle('Simple')
        self.q1 = QPixmap(filename1)
        self.q2 = QPixmap(filename2)
        bt = QPushButton('Compute', self)
        bt.setGeometry(xx / 2 - 50, p2.shape[1] - 10, 100, 25)
        bt.clicked.connect(self.startcalc)
        self.ispainter = False
        self.ptx = []
        self.pty = []
        self.mixed = False

    def solveEqu(self, cnt, equ, k):
        A = [[0 for j in range(0, cnt)] for i in range(0, cnt)]
        B = [0 for j in range(0, cnt)]
        for i in range(0, cnt):
            A[i] = equ[i][0:cnt]
            B[i] = equ[i][cnt][k]
        A = np.array(A)
        B = np.array(B)
        A = scipy.sparse.csr_matrix(A.astype(float))
        x = scipy.sparse.linalg.spsolve(A, B)
        return np.array(x)

    def startcalc(self):
        typ = [[0 for j in range(0, m)] for i in range(0, n)]
        cat = [[0 for j in range(0, m)] for i in range(0, n)]
        pos = []
        cnt = 0
        for (ptx, pty) in zip(self.ptx, self.pty):
            for i in range(ptx, ptx + 10):
                for j in range(pty, pty + 10):
                    if i - 10 < m and j < n:
                        typ[j][i - 10] = 1
        for i in range(0, n):
            for j in range(0, m):
                if typ[i][j] == 1:
                    cat[i][j] = cnt
                    cnt += 1
                    pos.append([i, j])
                    if i == 0 or i == n-1 or j == 0 or j == m-1 or typ[i-1][j] == 0 or typ[i+1][j] == 0 or typ[i][j+1] == 0 or typ[i][j-1] == 0:
                        typ[i][j] = 2

        equ = [[0 for j in range(0, cnt + 1)] for i in range(0, cnt)]
        dx = [0, 0, -1, 1]
        dy = [-1, 1, 0, 0]
        for i in range(0, cnt):
            x, y = pos[i]
            npp = 0
            if typ[x][y] == 1:
                for j in range(0, 4):
                    nx, ny = x + dx[j], y + dy[j]
                    if nx >= 0 and nx < n and ny >= 0 and ny < m:
                        npp += 1
                        if typ[nx][ny] == 1:
                            equ[i][cat[nx][ny]] = -1
                        elif typ[nx][ny] == 2:
                            equ[i][cnt] += self.p2[nx][ny]
                        # Here's the realization of h_pq
                        f_pq = self.p1[x][y] - self.p1[nx][ny]
                        g_pq = self.p2[x][y] - self.p2[nx][ny]
                        if np.dot(f_pq, f_pq) > np.dot(g_pq, g_pq):
                            h_pq = f_pq
                        else:
                            h_pq = g_pq
                        equ[i][cnt] += h_pq
                equ[i][i] = npp
            elif typ[x][y] == 2:
                equ[i][i] = 1
                equ[i][cnt] = self.p2[x][y]
        equ = np.array(equ)
        ans = np.array(self.p2)
        xs = self.solveEqu(cnt, equ, 0)
        ys = self.solveEqu(cnt, equ, 1)
        zs = self.solveEqu(cnt, equ, 2)
        for i in range(0, cnt):
            x, y = pos[i]
            xx = [xs[i], ys[i], zs[i], 1.0]
            for k in range(0, 3):
                if xx[k] < 0:
                    xx[k] = 0
                if xx[k] > 1:
                    xx[k] = 1
            xx = np.array(xx)
            ans[x][y] = xx

        """for i in range(0, cnt):
            x, y = pos[i]
            xx = equ[i]
            for nxt in range(i + 1, cnt):
                pq = equ[nxt][i] / equ[i][i]
                if np.abs(pq) > 0.0001:
                    equ[nxt] -= pq * xx
        f=open('a.txt', 'w')
        f.write(str(list(equ)))
        f.close()
        for i in range(cnt-1, -1, -1):
            for j in range(i+1, cnt):
                equ[i][cnt] -= equ[j][cnt] * equ[i][j]
            equ[i][cnt] /= equ[i][i]
            x, y = pos[i]
            xx = equ[i][cnt]
            for k in range(0, 3):
                if xx[k] < 0:
                    xx[k] = 0
                if xx[k] > 1:
                    xx[k] = 1
            xx[3] = 1
            if x < self.p2.shape[1] and y < self.p2.shape[0]:
                ans[y][x] = xx"""

        matplotlib.image.imsave('save.png', ans)
        self.mixed = True
        self.repaint()

    def paintEvent(self, event):
        self.p = QPainter(self)
        self.p.drawPixmap(10, 0, self.p1.shape[1], self.p1.shape[0], self.q1)
        self.p.drawPixmap(p1.shape[1] + 40, 0, self.p2.shape[1], self.p2.shape[0], self.q2)
        if self.mixed:
            qq = QPixmap('save.png')
            self.p.drawPixmap(self.xx / 2 - self.p2.shape[1]/2, p2.shape[1] + 30, self.p2.shape[1], self.p2.shape[0], qq)
        self.p.setPen(QPen(QColor(0, 160, 230, 0), 1))
        self.p.setBrush(QBrush(QColor(0, 160, 230, 10)))
        for (ptx, pty) in zip(self.ptx, self.pty):
            self.p.drawRect(ptx, pty, 10, 10)
        self.p.setPen(QPen(QColor(0, 160, 230, 0), 1))
        self.p.setBrush(QBrush(QColor(0, 160, 230, 10)))
        for (ptx, pty) in zip(self.ptx, self.pty):
            self.p.drawRect(ptx + p1.shape[1] + 30, pty, 10, 10)
        self.p.end()

    def mousePressEvent(self, event):
        ptx = event.x()
        pty = event.y()
        self.ispainter = True
        if ptx < 10 or ptx > 10 + p1.shape[1] or pty < 0 or pty > p1.shape[0]:
            return
        self.ptx.append(ptx)
        self.pty.append(pty)
        self.update()

    def mouseMoveEvent(self, event):
        if not self.ispainter:
            return
        ptx = event.x()
        pty = event.y()
        if ptx < 10 or ptx > 10 + p1.shape[1] or pty < 0 or pty > p1.shape[0]:
            return
        self.ptx.append(ptx)
        self.pty.append(pty)
        self.update()

    def mouseReleaseEvent(self, event):
        self.ispainter = False
        self.update()

filename1 = 'p1.png'
filename2 = 'p2.png'

p1 = np.array(pyl.imread(filename1))
p2 = np.array(pyl.imread(filename2))
print(p1.shape)
print(p2.shape)

n = min(p1.shape[0], p2.shape[0])
m = min(p1.shape[1], p2.shape[1])

app = QApplication(sys.argv)
w = Mywindow(p1, p2, n, m)

w.show()

sys.exit(app.exec_())

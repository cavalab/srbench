import time

from sympy import *
import numpy as np
from sklearn.metrics import mean_squared_error
import timeout_decorator
import copy
import itertools

CountACC = 0.0


def Global():
    global CountACC


x, y, z, v, w, a, b, c, d = symbols("x,y,z,v,w,a,b,c,d")
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 = symbols(
    "x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29 ")


class Point:
    name = 'Select k+1 Points to calculate Taylor Series'

    def __init__(self, in1=0, in2=0, in3=0, in4=0, in5=0, target=0, expansionPoint=2., varNum=1):
        self.varNum = varNum
        self.in1 = in1
        self.in2 = in2
        self.in3 = in3
        self.in4 = in4
        self.in5 = in5
        self.target = target
        self.expansionPoint = expansionPoint

    def __lt__(self, other):
        if self.varNum == 1:
            return abs(self.in1 - self.expansionPoint) < abs(other.in1 - self.expansionPoint)
        elif self.varNum == 2:
            return (self.in1 - self.expansionPoint) ** 2 + (self.in2 - self.expansionPoint) ** 2 < (
                    other.in1 - self.expansionPoint) ** 2 + (other.in2 - self.expansionPoint) ** 2
        elif self.varNum == 3:
            return (self.in1 - self.expansionPoint) ** 2 + (self.in2 - self.expansionPoint) ** 2 + (
                    self.in3 - self.expansionPoint) ** 2 < (other.in1 - self.expansionPoint) ** 2 + (
                           other.in2 - self.expansionPoint) ** 2 + (other.in3 - self.expansionPoint) ** 2
        elif self.varNum == 4:
            return (self.in1 - self.expansionPoint) ** 2 + (self.in2 - self.expansionPoint) ** 2 + (
                    self.in3 - self.expansionPoint) ** 2 + (self.in4 - self.expansionPoint) ** 2 < (
                           other.in1 - self.expansionPoint) ** 2 + (other.in2 - self.expansionPoint) ** 2 + (
                           other.in3 - self.expansionPoint) ** 2 + (other.in4 - self.expansionPoint) ** 2
        elif self.varNum == 5:
            return (self.in1 - self.expansionPoint) ** 2 + (self.in2 - self.expansionPoint) ** 2 + (
                    self.in3 - self.expansionPoint) ** 2 + (self.in4 - self.expansionPoint) ** 2 + (
                           self.in5 - self.expansionPoint) ** 2 < (other.in1 - self.expansionPoint) ** 2 + (
                           other.in2 - self.expansionPoint) ** 2 + (other.in3 - self.expansionPoint) ** 2 + (
                           other.in4 - self.expansionPoint) ** 2 + (other.in5 - self.expansionPoint) ** 2


class Metrics:
    name = 'Good calculator'

    def __init__(self, fileName=0,dataSet =None, model=None, f=None, classNum=8, varNum=1):
        self.model = model
        self.f_taylor = 0
        self.f_low_taylor = 0
        self.fileName = fileName
        self.dataSet = dataSet
        self.classNum = classNum
        self.x, self.y, self.z, self.v, self.w = symbols("x,y,z,v,w")
        self.x0, self.x1, self.x2, self.x3, self.x4 = symbols("x0,x1,x2,x3,x4")
        _x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
        self._x = _x[:varNum]
        self.count = 0
        self.mediumcount = 0
        self.supercount = 0
        self.count1 = 0
        self.mediumcount1 = 0
        self.supercount1 = 0
        self.count2 = 0
        self.mediumcount2 = 0
        self.supercount2 = 0
        self.tempVector = np.zeros((1, 6, 126))
        self.varNum = varNum
        self.di_jian_flag = False
        self.parity_flag = False
        self.bias = 0.
        self.nmse = float("inf")
        self.low_nmse = float("inf")
        self.mse_log = float("inf")
        self.Y_log = None
        self.b_log = None
        self.Taylor_log = None
        self.f_taylor_log = None
        self.A = None
        self.midpoint = None
        self.Y_left, self.Y_right = None, None
        self.X_left, self.X_right = None, None
        X_Y = dataSet
        self.expantionPoint = copy.deepcopy(X_Y[0])
        self.mmm = X_Y.shape[0] - 1
        np.random.shuffle(X_Y)
        change = True
        for i in range(self.mmm):
            if (X_Y[i] == self.expantionPoint).all():
                X_Y[[i, -1], :] = X_Y[[-1, i], :]
                break

        X, Y = np.split(X_Y, (-1,), axis=1)
        self.X = copy.deepcopy(X)
        _X = []
        len = X.shape[1]
        for i in range(len):
            X, temp = np.split(X, (-1,), axis=1)
            temp = temp.reshape(-1)
            _X.extend([temp])
        _X.reverse()
        self._X, self.X_Y, self.Y = _X, X_Y, Y.reshape(-1)
        self.f0_log, self.Y_log = np.log(X_Y[0][-1]), np.log(abs(self.Y))
        self.b, self.b_log = (self.Y - self.expantionPoint[-1])[:-1], (self.Y_log - self.f0_log)[:-1]
        self.nihe_flag = False
        self._mid_left, self._mid_right = 0, 0
        self._x_left, self._x_right = 0, 0
        # try:
        if varNum == 1:
            self.taylor, self.expantionPointa0, self.expantionPointf0, self.X0, self.Y = self._getData_1var()
            self._X = [self.X0]
        elif varNum == 2:
            self.taylor = self._getData_2var()
        elif varNum == 3:
            # self.taylor = self._getData_3var()
            self.taylor = self._getData_xvar(3)
        elif varNum == 4:
            # self.taylor = self._getData_4var()
            self.taylor = self._getData_xvar(4)

        elif varNum == 5:
            # self.taylor = self._getData_5var()
            self.taylor = self._getData_xvar(5)
        elif varNum == 6:
            self.taylor = self._getData_6var()
        else:
            self.taylor = np.array([1] * 10000)
        # except BaseException:
        #     print('metrix error')
        #     self.taylor = np.array([1] * 10000)
        self.f_taylor = self._getTaylorPolynomial(varNum=varNum)

    def _getData_1var(self, k=18, taylorNum=18):

        mmm = self.X.shape[0] - 1
        b = [0.0] * mmm
        X_Y = self.dataSet
        a0 = X_Y[int(X_Y.shape[0] / 2)][0]
        f0 = X_Y[int(X_Y.shape[0] / 2)][1]
        np.random.shuffle(X_Y)
        X, Y = np.hsplit(X_Y, 2)
        for i in range(mmm):  #
            if X[i] == a0:
                X[[i, -1], :] = X[[-1, i], :]  #
                Y[[i, -1], :] = Y[[-1, i], :]  #
                break
        self.X = X
        X, Y = X.reshape(-1), Y.reshape(-1)  #
        self.Y = Y
        A = np.zeros((mmm, mmm))
        for i in range(mmm):
            b[i] = Y[i] - f0

        for i in range(mmm):
            for j in range(mmm):
                A[i][j] = ((X[i] - a0) ** (j + 1))
        Taylor = np.linalg.solve(A, b)
        Taylor = np.insert(Taylor, 0, f0)  #
        return Taylor.tolist()[:taylorNum], a0, f0, X, Y

    def _getData_2var(self, k=5, taylorNum=91):

        n = 2  #
        m = 1000
        mmm = self.X.shape[0] - 1

        A = None
        _x0 = self._X[0][:-1] - self.expantionPoint[0]
        _x1 = self._X[1][:-1] - self.expantionPoint[1]
        tempCount = 0
        flag = true
        for i in range(1, m + 1):  #
            if flag == True:
                for j in range(0, i + 1):  #
                    if A is None:
                        A = ((_x0 ** (i - j)) * (_x1 ** j)).reshape(mmm, 1)
                    else:
                        A = np.hstack((A, ((_x0 ** (i - j)) * (_x1 ** j)).reshape(mmm, 1)))
                    tempCount += 1
                    if tempCount == mmm:
                        flag = false
                        break
            else:
                break

        Taylor = np.linalg.solve(A, self.b)
        Taylor_log = np.linalg.solve(A, self.b_log)
        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])  #
        self.Taylor_log = np.insert(Taylor_log, 0, self.f0_log)[:taylorNum]
        self.A = A
        return Taylor.tolist()[:taylorNum]
#yxGao
    def _getData_xvar(self, n):
        start = time.time()
        def generate_tri(amount):
            triang = np.ones((amount, amount), dtype=int)
            for i in range(2, triang.shape[0]):
                triang[i][1:i] = triang[i - 1][1:i] + triang[i - 1][0:i - 1]
            return triang

        def combox(nn, m, tri):
            return tri[nn][min(m, nn - m)]

        def get_data(n, lines, matrix):
            tri = generate_tri(40)  #
            pre_sum = 0
            cur_sum = combox(n, n - 1, tri)  # init C (n + k - 1 , n - 1)
            cur_k = 1
            ret = np.ones((lines, lines))
            for i in range(lines):
                mark = -1
                if i < cur_sum:
                    mark = i - pre_sum
                elif i == cur_sum:
                    cur_k += 1
                    pre_sum = cur_sum
                    cur_sum += combox(cur_k + n - 1, n - 1, tri)
                    mark = 0
                else:
                    print('error')
                    # error
                # make cur_k divide into n combox(cur_k + n - 1, n - 1, tri)
                c = np.zeros(n, dtype=int)

                cur_idx = 1
                cp_mk = mark
                for j in range(n - 1):
                    while mark >= combox(cur_k + n - 1 - cur_idx, n - 1 - (j + 1), tri):
                        mark -= combox(cur_k + n - 1 - cur_idx, n - 1 - (j + 1), tri)
                        cur_idx += 1
                    c[j + 1] = cur_idx
                    cur_idx += 1
                # c[n] = cur_k - np.sum(c[0:n-1])
                res = np.zeros(n, dtype=int)

                res[:-1] = c[1:] - c[:-1] - 1
                # if ( 1 ,2 ,3 ) is selected from
                res[n - 1] = cur_k - np.sum(res[0:n - 1])
                # print(cur_k,res)
                for j in range(n):
                    ret[i][0:] *= matrix[j][0:] ** res[n - 1 - j]
            print(cur_k)
            return ret.transpose()

        mmm = self.X.shape[0] - 1  #
        X = np.zeros((n, mmm))
        for i in range(n):
            X[i] = self._X[i][:mmm] - self.expantionPoint[i]
        # @yxgao taylorNum = ?
        # @yxgao m = ?
        A = get_data(n, mmm, X)
        Taylor = np.linalg.solve(A, self.b)
        # Taylor_log = np.linalg.solve(A, self.b_log)
        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])  #
        # self.Taylor_log = np.insert(Taylor_log, 0, self.f0_log)[:]
        self.A = A
        end = time.time()
        print("time=",end-start)
        if n==3:
            TaylorNum = 455
        elif n==4:
            TaylorNum = 1820
        else:
            TaylorNum = 6188
        return Taylor.tolist()[:TaylorNum]#TaylorNum后期再改

    def _getData_3var(self, k=8, taylorNum=455):  #
        n = 3  #
        m = 3000
        mmm = self.X.shape[0] - 1  #
        A = None
        _x0 = self._X[0][:-1] - self.expantionPoint[0]
        _x1 = self._X[1][:-1] - self.expantionPoint[1]
        _x2 = self._X[2][:-1] - self.expantionPoint[2]
        tempCount = 0
        flag = true
        for k in range(1, m + 1):
            if flag == True:
                for c1 in range(0, k + 1):
                    if flag == True:
                        for c2 in range(0, k + 1):
                            if c1 + c2 <= k:
                                c3 = k - c1 - c2
                                if A is None:
                                    A = ((_x0 ** c3) * (_x1 ** c2) * (_x2 ** c1)).reshape(mmm, 1)
                                else:
                                    A = np.hstack((A, (((_x0 ** c3) * (_x1 ** c2) * (_x2 ** c1)).reshape(mmm, 1))))
                                tempCount += 1
                                if tempCount == mmm:
                                    flag = false
                                    break

        Taylor = np.linalg.solve(A, self.b)
        Taylor_log = np.linalg.solve(A, self.b_log)
        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])  #
        self.Taylor_log = np.insert(Taylor_log, 0, self.f0_log)[:taylorNum]
        self.A = A
        return Taylor.tolist()[:taylorNum]

    def _getData_4var(self, k=8, taylorNum=1820):
        n = 4  #
        m = 8000
        mmm = self.X.shape[0] - 1  #

        A = None
        B = None
        C = None
        D = None
        _x0 = self._X[0][:-1] - self.expantionPoint[0]
        _x1 = self._X[1][:-1] - self.expantionPoint[1]
        _x2 = self._X[2][:-1] - self.expantionPoint[2]
        _x3 = self._X[3][:-1] - self.expantionPoint[3]

        tempCount = 0
        for k in range(1, m + 1):
            for c1 in range(0, k + 1):
                for c2 in range(0, k + 1):
                    for c3 in range(0, k + 1):
                        if c1 + c2 + c3 <= k:
                            c4 = k - c1 - c2 - c3
                            if tempCount < 1000:
                                if A is None:
                                    A = ((_x0 ** c4) * (_x1 ** c3) * (_x2 ** c2) * (_x3 ** c1)).reshape(mmm, 1)
                                else:
                                    A = np.hstack(
                                        (A, ((_x0 ** c4) * (_x1 ** c3) * (_x2 ** c2) * (_x3 ** c1)).reshape(mmm, 1)))
                            elif tempCount < 2000:
                                if B is None:
                                    B = ((_x0 ** c4) * (_x1 ** c3) * (_x2 ** c2) * (_x3 ** c1)).reshape(mmm,
                                                                                                        1)
                                else:
                                    B = np.hstack((B, ((_x0 ** c4) * (_x1 ** c3) * (_x2 ** c2) * (
                                            _x3 ** c1)).reshape(mmm, 1)))
                            elif tempCount < 3000:
                                if C is None:
                                    C = ((_x0 ** c4) * (_x1 ** c3) * (_x2 ** c2) * (_x3 ** c1)).reshape(mmm,
                                                                                                        1)
                                else:
                                    C = np.hstack((C, ((_x0 ** c4) * (_x1 ** c3) * (_x2 ** c2) * (
                                            _x3 ** c1)).reshape(mmm, 1)))
                            else:
                                if D is None:
                                    D = ((_x0 ** c4) * (_x1 ** c3) * (_x2 ** c2) * (_x3 ** c1)).reshape(mmm,
                                                                                                        1)
                                else:
                                    D = np.hstack((D, ((_x0 ** c4) * (_x1 ** c3) * (_x2 ** c2) * (
                                            _x3 ** c1)).reshape(mmm, 1)))
                            tempCount += 1
                            if tempCount == mmm:
                                break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        if B is not None:
            A = np.hstack((A, B))
        if C is not None:
            A = np.hstack((A, C))
        if D is not None:
            A = np.hstack((A, D))
        Taylor = np.linalg.solve(A, self.b)
        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])
        return Taylor.tolist()[:taylorNum]

    def _getData_5var(self, k=5, taylorNum=6188):
        n = 5  #
        m = 10000
        mmm = self.X.shape[0] - 1
        A = None
        B = None
        C = None
        D = None
        _x0 = self._X[0][:-1] - self.expantionPoint[0]
        _x1 = self._X[1][:-1] - self.expantionPoint[1]
        _x2 = self._X[2][:-1] - self.expantionPoint[2]
        _x3 = self._X[3][:-1] - self.expantionPoint[3]
        _x4 = self._X[4][:-1] - self.expantionPoint[4]

        tempCount = 0
        for k in range(1, m + 1):
            for c1 in range(0, k + 1):
                for c2 in range(0, k + 1):
                    for c3 in range(0, k + 1):
                        for c4 in range(0, k + 1):
                            if c1 + c2 + c3 + c4 <= k:
                                c5 = k - c1 - c2 - c3 - c4
                                if tempCount < 2000:
                                    if A is None:
                                        A = ((_x0 ** c5) * \
                                             (_x1 ** c4) * \
                                             (_x2 ** c3) * \
                                             (_x3 ** c2) * \
                                             (_x4 ** c1)).reshape(mmm, 1)
                                    else:
                                        A = np.hstack(
                                            (A, ((_x0 ** c5) * \
                                                 (_x1 ** c4) * \
                                                 (_x2 ** c3) * \
                                                 (_x3 ** c2) * \
                                                 (_x4 ** c1)).reshape(mmm, 1)))
                                elif tempCount < 4000:
                                    if B is None:
                                        B = ((_x0 ** c5) * \
                                             (_x1 ** c4) * \
                                             (_x2 ** c3) * \
                                             (_x3 ** c2) * \
                                             (_x4 ** c1)).reshape(mmm, 1)
                                    else:
                                        B = np.hstack(
                                            (B, ((_x0 ** c5) * \
                                                 (_x1 ** c4) * \
                                                 (_x2 ** c3) * \
                                                 (_x3 ** c2) * \
                                                 (_x4 ** c1)).reshape(mmm, 1)))
                                elif tempCount < 6000:
                                    if C is None:
                                        C = ((_x0 ** c5) * \
                                             (_x1 ** c4) * \
                                             (_x2 ** c3) * \
                                             (_x3 ** c2) * \
                                             (_x4 ** c1)).reshape(mmm, 1)
                                    else:
                                        C = np.hstack(
                                            (C, ((_x0 ** c5) * \
                                                 (_x1 ** c4) * \
                                                 (_x2 ** c3) * \
                                                 (_x3 ** c2) * \
                                                 (_x4 ** c1)).reshape(mmm, 1)))
                                else:
                                    if D is None:
                                        D = ((_x0 ** c5) * \
                                             (_x1 ** c4) * \
                                             (_x2 ** c3) * \
                                             (_x3 ** c2) * \
                                             (_x4 ** c1)).reshape(mmm, 1)
                                    else:
                                        D = np.hstack(
                                            (D, ((_x0 ** c5) * \
                                                 (_x1 ** c4) * \
                                                 (_x2 ** c3) * \
                                                 (_x3 ** c2) * \
                                                 (_x4 ** c1)).reshape(mmm, 1)))
                                tempCount += 1
                                if tempCount == mmm:
                                    break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        if B is not None:
            A = np.hstack((A, B))
        if C is not None:
            A = np.hstack((A, C))
        if D is not None:
            A = np.hstack((A, D))

        Taylor = np.linalg.solve(A, self.b)
        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])
        return Taylor.tolist()[:taylorNum]

    def _getData_6var(self, k=5, taylorNum=3003):
        n = 6  #
        m = 10000
        mmm = self.X.shape[0] - 1

        A = None
        B = None
        C = None
        D = None
        _x0 = self._X[0][:-1] - self.expantionPoint[0]
        _x1 = self._X[1][:-1] - self.expantionPoint[1]
        _x2 = self._X[2][:-1] - self.expantionPoint[2]
        _x3 = self._X[3][:-1] - self.expantionPoint[3]
        _x4 = self._X[4][:-1] - self.expantionPoint[4]
        _x5 = self._X[5][:-1] - self.expantionPoint[5]

        tempCount = 0
        for k in range(1, m + 1):
            for c1 in range(0, k + 1):
                for c2 in range(0, k + 1):
                    for c3 in range(0, k + 1):
                        for c4 in range(0, k + 1):
                            for c5 in range(0, k + 1):
                                if c1 + c2 + c3 + c4 + c5 <= k:
                                    c6 = k - c1 - c2 - c3 - c4 - c5
                                    if tempCount < 2000:
                                        if A is None:
                                            A = ((_x0 ** c6) * \
                                                 (_x1 ** c5) * \
                                                 (_x2 ** c4) * \
                                                 (_x3 ** c3) * \
                                                 (_x4 ** c2) * \
                                                 (_x5 ** c1)).reshape(mmm, 1)
                                        else:
                                            A = np.hstack((A,
                                                           ((_x0 ** c6) * \
                                                            (_x1 ** c5) * \
                                                            (_x2 ** c4) * \
                                                            (_x3 ** c3) * \
                                                            (_x4 ** c2) * \
                                                            (_x5 ** c1)).reshape(mmm, 1)))
                                    elif tempCount < 4000:
                                        if B is None:
                                            B = ((_x0 ** c6) * \
                                                 (_x1 ** c5) * \
                                                 (_x2 ** c4) * \
                                                 (_x3 ** c3) * \
                                                 (_x4 ** c2) * \
                                                 (_x5 ** c1)).reshape(mmm, 1)
                                        else:
                                            B = np.hstack((B,
                                                           ((_x0 ** c6) * \
                                                            (_x1 ** c5) * \
                                                            (_x2 ** c4) * \
                                                            (_x3 ** c3) * \
                                                            (_x4 ** c2) * \
                                                            (_x5 ** c1)).reshape(mmm, 1)))
                                    elif tempCount < 6000:
                                        if C is None:
                                            C = ((_x0 ** c6) * \
                                                 (_x1 ** c5) * \
                                                 (_x2 ** c4) * \
                                                 (_x3 ** c3) * \
                                                 (_x4 ** c2) * \
                                                 (_x5 ** c1)).reshape(mmm, 1)
                                        else:
                                            C = np.hstack((C,
                                                           ((_x0 ** c6) * \
                                                            (_x1 ** c5) * \
                                                            (_x2 ** c4) * \
                                                            (_x3 ** c3) * \
                                                            (_x4 ** c2) * \
                                                            (_x5 ** c1)).reshape(mmm, 1)))
                                    else:
                                        if D is None:
                                            D = ((_x0 ** c6) * \
                                                 (_x1 ** c5) * \
                                                 (_x2 ** c4) * \
                                                 (_x3 ** c3) * \
                                                 (_x4 ** c2) * \
                                                 (_x5 ** c1)).reshape(mmm, 1)
                                        else:
                                            D = np.hstack((D,
                                                           ((_x0 ** c6) * \
                                                            (_x1 ** c5) * \
                                                            (_x2 ** c4) * \
                                                            (_x3 ** c3) * \
                                                            (_x4 ** c2) * \
                                                            (_x5 ** c1)).reshape(mmm, 1)))
                                    tempCount += 1
                                    if tempCount == mmm:
                                        break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        if B is not None:
            A = np.hstack((A, B))
        if C is not None:
            A = np.hstack((A, C))
        if D is not None:
            A = np.hstack((A, D))
        Taylor = np.linalg.solve(A, self.b)
        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])  #
        return Taylor.tolist()[:taylorNum]

    def _getData_8var(self, k=5, taylorNum=3003):  #
        n = 8  #
        m = 10000
        mmm = self.X.shape[0] - 1  #
        print('Matrix size', mmm)

        A = None
        B = None
        C = None
        D = None
        _x0 = self._X[0][:-1] - self.expantionPoint[0]
        _x1 = self._X[1][:-1] - self.expantionPoint[1]
        _x2 = self._X[2][:-1] - self.expantionPoint[2]
        _x3 = self._X[3][:-1] - self.expantionPoint[3]
        _x4 = self._X[4][:-1] - self.expantionPoint[4]
        _x5 = self._X[5][:-1] - self.expantionPoint[5]
        _x6 = self._X[6][:-1] - self.expantionPoint[6]
        _x7 = self._X[7][:-1] - self.expantionPoint[7]

        tempCount = 0
        for k in range(1, m + 1):
            for c1 in range(0, k + 1):
                for c2 in range(0, k + 1):
                    for c3 in range(0, k + 1):
                        for c4 in range(0, k + 1):
                            for c5 in range(0, k + 1):
                                for c6 in range(0, k + 1):
                                    for c7 in range(0, k + 1):
                                        if c1 + c2 + c3 + c4 + c5 + c6 + c7 <= k:
                                            c8 = k - c1 - c2 - c3 - c4 - c5 - c6 - c7
                                            if tempCount < 2000:
                                                if A is None:
                                                    A = ((_x0 ** c8) * \
                                                         (_x1 ** c7) * \
                                                         (_x2 ** c6) * \
                                                         (_x3 ** c5) * \
                                                         (_x4 ** c4) * \
                                                         (_x5 ** c3) * \
                                                         (_x6 ** c2) * \
                                                         (_x7 ** c1)).reshape(mmm, 1)
                                                else:
                                                    A = np.hstack((A, ((_x0 ** c8) * \
                                                                       (_x1 ** c7) * \
                                                                       (_x2 ** c6) * \
                                                                       (_x3 ** c5) * \
                                                                       (_x4 ** c4) * \
                                                                       (_x5 ** c3) * \
                                                                       (_x6 ** c2) * \
                                                                       (_x7 ** c1)).reshape(mmm, 1)))
                                            elif tempCount < 4000:
                                                if B is None:
                                                    B = ((_x0 ** c8) * \
                                                         (_x1 ** c7) * \
                                                         (_x2 ** c6) * \
                                                         (_x3 ** c5) * \
                                                         (_x4 ** c4) * \
                                                         (_x5 ** c3) * \
                                                         (_x6 ** c2) * \
                                                         (_x7 ** c1)).reshape(mmm, 1)
                                                else:
                                                    B = np.hstack((B, ((_x0 ** c8) * \
                                                                       (_x1 ** c7) * \
                                                                       (_x2 ** c6) * \
                                                                       (_x3 ** c5) * \
                                                                       (_x4 ** c4) * \
                                                                       (_x5 ** c3) * \
                                                                       (_x6 ** c2) * \
                                                                       (_x7 ** c1)).reshape(mmm, 1)))
                                            elif tempCount < 6000:
                                                if C is None:
                                                    C = ((_x0 ** c8) * \
                                                         (_x1 ** c7) * \
                                                         (_x2 ** c6) * \
                                                         (_x3 ** c5) * \
                                                         (_x4 ** c4) * \
                                                         (_x5 ** c3) * \
                                                         (_x6 ** c2) * \
                                                         (_x7 ** c1)).reshape(mmm, 1)
                                                else:
                                                    C = np.hstack((C, ((_x0 ** c8) * \
                                                                       (_x1 ** c7) * \
                                                                       (_x2 ** c6) * \
                                                                       (_x3 ** c5) * \
                                                                       (_x4 ** c4) * \
                                                                       (_x5 ** c3) * \
                                                                       (_x6 ** c2) * \
                                                                       (_x7 ** c1)).reshape(mmm, 1)))
                                            else:
                                                if D is None:
                                                    D = ((_x0 ** c8) * \
                                                         (_x1 ** c7) * \
                                                         (_x2 ** c6) * \
                                                         (_x3 ** c5) * \
                                                         (_x4 ** c4) * \
                                                         (_x5 ** c3) * \
                                                         (_x6 ** c2) * \
                                                         (_x7 ** c1)).reshape(mmm, 1)
                                                else:
                                                    D = np.hstack((D, ((_x0 ** c8) * \
                                                                       (_x1 ** c7) * \
                                                                       (_x2 ** c6) * \
                                                                       (_x3 ** c5) * \
                                                                       (_x4 ** c4) * \
                                                                       (_x5 ** c3) * \
                                                                       (_x6 ** c2) * \
                                                                       (_x7 ** c1)).reshape(mmm, 1)))
                                            tempCount += 1
                                            if tempCount == mmm:
                                                break
                                    else:
                                        continue
                                    break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        if B is not None:
            A = np.hstack((A, B))
        if C is not None:
            A = np.hstack((A, C))
        if D is not None:
            A = np.hstack((A, D))
        Taylor = np.linalg.solve(A, self.b)
        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])
        return Taylor.tolist()[:taylorNum]  #

    def _getData_9var(self, k=5, taylorNum=5005):  #
        n = 8  #
        m = 5835
        mmm = self.X.shape[0] - 1
        print('Matrix size', mmm)

        A = None
        B = None
        C = None
        D = None
        _x0 = self._X[0][:-1] - self.expantionPoint[0]
        _x1 = self._X[1][:-1] - self.expantionPoint[1]
        _x2 = self._X[2][:-1] - self.expantionPoint[2]
        _x3 = self._X[3][:-1] - self.expantionPoint[3]
        _x4 = self._X[4][:-1] - self.expantionPoint[4]
        _x5 = self._X[5][:-1] - self.expantionPoint[5]
        _x6 = self._X[6][:-1] - self.expantionPoint[6]
        _x7 = self._X[7][:-1] - self.expantionPoint[7]
        _x8 = self._X[8][:-1] - self.expantionPoint[8]
        '''
        for cc1 in range(mmm):
            _x0 = self._X[0][cc1] - self.expantionPoint[0]
            _x1 = self._X[1][cc1] - self.expantionPoint[1]
            _x2 = self._X[2][cc1] - self.expantionPoint[2]
            _x3 = self._X[3][cc1] - self.expantionPoint[3]
            _x4 = self._X[4][cc1] - self.expantionPoint[4]
            _x5 = self._X[5][cc1] - self.expantionPoint[5]
            _x6 = self._X[6][cc1] - self.expantionPoint[6]
            _x7 = self._X[7][cc1] - self.expantionPoint[7]
            _x8 = self._X[8][cc1] - self.expantionPoint[8]
        '''
        tempCount = 0
        for k in range(1, m + 1):
            for c1 in range(0, k + 1):
                for c2 in range(0, k + 1):
                    for c3 in range(0, k + 1):
                        for c4 in range(0, k + 1):
                            for c5 in range(0, k + 1):
                                for c6 in range(0, k + 1):
                                    for c7 in range(0, k + 1):
                                        for c8 in range(0, k + 1):
                                            if c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 <= k:
                                                c9 = k - c1 - c2 - c3 - c4 - c5 - c6 - c7 - c8
                                                if tempCount < 2000:
                                                    if A is None:
                                                        A = ((_x0 ** c9) * \
                                                             (_x1 ** c8) * \
                                                             (_x2 ** c7) * \
                                                             (_x3 ** c6) * \
                                                             (_x4 ** c5) * \
                                                             (_x5 ** c4) * \
                                                             (_x6 ** c3) * \
                                                             (_x7 ** c2) * \
                                                             (_x8 ** c1)).reshape(mmm, 1)
                                                    else:
                                                        A = np.hstack((A, ((_x0 ** c9) * \
                                                                           (_x1 ** c8) * \
                                                                           (_x2 ** c7) * \
                                                                           (_x3 ** c6) * \
                                                                           (_x4 ** c5) * \
                                                                           (_x5 ** c4) * \
                                                                           (_x6 ** c3) * \
                                                                           (_x7 ** c2) * \
                                                                           (_x8 ** c1)).reshape(mmm, 1)))
                                                elif tempCount < 4000:
                                                    if B is None:
                                                        B = ((_x0 ** c9) * \
                                                             (_x1 ** c8) * \
                                                             (_x2 ** c7) * \
                                                             (_x3 ** c6) * \
                                                             (_x4 ** c5) * \
                                                             (_x5 ** c4) * \
                                                             (_x6 ** c3) * \
                                                             (_x7 ** c2) * \
                                                             (_x8 ** c1)).reshape(mmm, 1)
                                                    else:
                                                        B = np.hstack((B, ((_x0 ** c9) * \
                                                                           (_x1 ** c8) * \
                                                                           (_x2 ** c7) * \
                                                                           (_x3 ** c6) * \
                                                                           (_x4 ** c5) * \
                                                                           (_x5 ** c4) * \
                                                                           (_x6 ** c3) * \
                                                                           (_x7 ** c2) * \
                                                                           (_x8 ** c1)).reshape(mmm, 1)))
                                                elif tempCount < 6000:
                                                    if C is None:
                                                        C = ((_x0 ** c9) * \
                                                             (_x1 ** c8) * \
                                                             (_x2 ** c7) * \
                                                             (_x3 ** c6) * \
                                                             (_x4 ** c5) * \
                                                             (_x5 ** c4) * \
                                                             (_x6 ** c3) * \
                                                             (_x7 ** c2) * \
                                                             (_x8 ** c1)).reshape(mmm, 1)
                                                    else:
                                                        C = np.hstack((C, ((_x0 ** c9) * \
                                                                           (_x1 ** c8) * \
                                                                           (_x2 ** c7) * \
                                                                           (_x3 ** c6) * \
                                                                           (_x4 ** c5) * \
                                                                           (_x5 ** c4) * \
                                                                           (_x6 ** c3) * \
                                                                           (_x7 ** c2) * \
                                                                           (_x8 ** c1)).reshape(mmm, 1)))
                                                else:
                                                    if D is None:
                                                        D = ((_x0 ** c9) * \
                                                             (_x1 ** c8) * \
                                                             (_x2 ** c7) * \
                                                             (_x3 ** c6) * \
                                                             (_x4 ** c5) * \
                                                             (_x5 ** c4) * \
                                                             (_x6 ** c3) * \
                                                             (_x7 ** c2) * \
                                                             (_x8 ** c1)).reshape(mmm, 1)
                                                    else:
                                                        D = np.hstack((D, ((_x0 ** c9) * \
                                                                           (_x1 ** c8) * \
                                                                           (_x2 ** c7) * \
                                                                           (_x3 ** c6) * \
                                                                           (_x4 ** c5) * \
                                                                           (_x5 ** c4) * \
                                                                           (_x6 ** c3) * \
                                                                           (_x7 ** c2) * \
                                                                           (_x8 ** c1)).reshape(mmm, 1)))
                                                tempCount += 1
                                                # print(tempCount)
                                                if tempCount == mmm:
                                                    break
                                        else:
                                            continue
                                        break
                                    else:
                                        continue
                                    break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        if B is not None:
            A = np.hstack((A, B))
        if C is not None:
            A = np.hstack((A, C))
        if D is not None:
            A = np.hstack((A, D))
        Taylor = np.linalg.solve(A, self.b)  # 求解k+2元一次方程
        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])  # 在开头加上0阶导
        return Taylor.tolist()[:taylorNum]  # 只保留0阶导+k阶导所有系数

    def _getData_10var(self, k=5, taylorNum=5005):  # 求解mmm元1次方程组
        # 展开至k阶k=mmm，共有Subject = m*(m+3)/2项，若要组成统一的矩阵，则需要再按5元最高阶的项数进行补0，先默认按5元展开至5阶最高阶一共128项
        n = 8  # 6元变量
        m = 10000  # m再大也不影响运行！！实际不需要展开至1000阶，但是展开至m阶的未知系数要比mmm大
        mmm = self.X.shape[0] - 1  # 除展开点有多少个样本点就全用上
        print('Matrix size', mmm)
        # A = np.zeros((mmm, mmm))
        # Count = 0  # 循环Count次
        A = None
        B = None
        C = None
        D = None
        E = None
        _x0 = self._X[0][:-1] - self.expantionPoint[0]
        _x1 = self._X[1][:-1] - self.expantionPoint[1]
        _x2 = self._X[2][:-1] - self.expantionPoint[2]
        _x3 = self._X[3][:-1] - self.expantionPoint[3]
        _x4 = self._X[4][:-1] - self.expantionPoint[4]
        _x5 = self._X[5][:-1] - self.expantionPoint[5]
        _x6 = self._X[6][:-1] - self.expantionPoint[6]
        _x7 = self._X[7][:-1] - self.expantionPoint[7]
        _x8 = self._X[8][:-1] - self.expantionPoint[8]
        _x9 = self._X[8][:-1] - self.expantionPoint[9]

        '''
        _x10 = self._X[10][:-1] - self.expantionPoint[10]
        _x11 = self._X[11][:-1] - self.expantionPoint[11]
        _x12 = self._X[12][:-1] - self.expantionPoint[12]
        _x13 = self._X[13][:-1] - self.expantionPoint[13]
        _x14 = self._X[14][:-1] - self.expantionPoint[14]
        _x15 = self._X[15][:-1] - self.expantionPoint[15]
        _x16 = self._X[16][:-1] - self.expantionPoint[16]
        _x17 = self._X[17][:-1] - self.expantionPoint[17]
        _x18 = self._X[18][:-1] - self.expantionPoint[18]
        _x19 = self._X[19][:-1] - self.expantionPoint[19]
        _x20 = self._X[20][:-1] - self.expantionPoint[20]
        '''
        tempCount = 0
        for k in range(1, m + 1):
            for c1 in range(0, k + 1):
                for c2 in range(0, k + 1):
                    for c3 in range(0, k + 1):
                        for c4 in range(0, k + 1):
                            for c5 in range(0, k + 1):
                                for c6 in range(0, k + 1):
                                    for c7 in range(0, k + 1):
                                        for c8 in range(0, k + 1):
                                            for c9 in range(0, k + 1):
                                                if c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 <= k:
                                                    c10 = k - c1 - c2 - c3 - c4 - c5 - c6 - c7 - c8 - c9
                                                    if tempCount < 2000:
                                                        if A is None:
                                                            A = ((_x0 ** c10) * \
                                                                 (_x1 ** c9) * \
                                                                 (_x2 ** c8) * \
                                                                 (_x3 ** c7) * \
                                                                 (_x4 ** c6) * \
                                                                 (_x5 ** c5) * \
                                                                 (_x6 ** c4) * \
                                                                 (_x7 ** c3) * \
                                                                 (_x8 ** c2) * \
                                                                 (_x9 ** c1)).reshape(mmm, 1)
                                                        else:
                                                            A = np.hstack((A, (((_x0 ** c10) * \
                                                                                (_x1 ** c9) * \
                                                                                (_x2 ** c8) * \
                                                                                (_x3 ** c7) * \
                                                                                (_x4 ** c6) * \
                                                                                (_x5 ** c5) * \
                                                                                (_x6 ** c4) * \
                                                                                (_x7 ** c3) * \
                                                                                (_x8 ** c2) * \
                                                                                (_x9 ** c1)).reshape(mmm, 1))))
                                                    elif tempCount < 4000:
                                                        if B is None:
                                                            B = ((_x0 ** c10) * \
                                                                 (_x1 ** c9) * \
                                                                 (_x2 ** c8) * \
                                                                 (_x3 ** c7) * \
                                                                 (_x4 ** c6) * \
                                                                 (_x5 ** c5) * \
                                                                 (_x6 ** c4) * \
                                                                 (_x7 ** c3) * \
                                                                 (_x8 ** c2) * \
                                                                 (_x9 ** c1)).reshape(mmm, 1)
                                                        else:
                                                            B = np.hstack((B, (((_x0 ** c10) * \
                                                                                (_x1 ** c9) * \
                                                                                (_x2 ** c8) * \
                                                                                (_x3 ** c7) * \
                                                                                (_x4 ** c6) * \
                                                                                (_x5 ** c5) * \
                                                                                (_x6 ** c4) * \
                                                                                (_x7 ** c3) * \
                                                                                (_x8 ** c2) * \
                                                                                (_x9 ** c1)).reshape(mmm, 1))))
                                                    elif tempCount < 6000:
                                                        if C is None:
                                                            C = ((_x0 ** c10) * \
                                                                 (_x1 ** c9) * \
                                                                 (_x2 ** c8) * \
                                                                 (_x3 ** c7) * \
                                                                 (_x4 ** c6) * \
                                                                 (_x5 ** c5) * \
                                                                 (_x6 ** c4) * \
                                                                 (_x7 ** c3) * \
                                                                 (_x8 ** c2) * \
                                                                 (_x9 ** c1)).reshape(mmm, 1)
                                                        else:
                                                            C = np.hstack((C, (((_x0 ** c10) * \
                                                                                (_x1 ** c9) * \
                                                                                (_x2 ** c8) * \
                                                                                (_x3 ** c7) * \
                                                                                (_x4 ** c6) * \
                                                                                (_x5 ** c5) * \
                                                                                (_x6 ** c4) * \
                                                                                (_x7 ** c3) * \
                                                                                (_x8 ** c2) * \
                                                                                (_x9 ** c1)).reshape(mmm, 1))))
                                                    else:
                                                        if D is None:
                                                            D = ((_x0 ** c10) * \
                                                                 (_x1 ** c9) * \
                                                                 (_x2 ** c8) * \
                                                                 (_x3 ** c7) * \
                                                                 (_x4 ** c6) * \
                                                                 (_x5 ** c5) * \
                                                                 (_x6 ** c4) * \
                                                                 (_x7 ** c3) * \
                                                                 (_x8 ** c2) * \
                                                                 (_x9 ** c1)).reshape(mmm, 1)
                                                        else:
                                                            D = np.hstack((D, (((_x0 ** c10) * \
                                                                                (_x1 ** c9) * \
                                                                                (_x2 ** c8) * \
                                                                                (_x3 ** c7) * \
                                                                                (_x4 ** c6) * \
                                                                                (_x5 ** c5) * \
                                                                                (_x6 ** c4) * \
                                                                                (_x7 ** c3) * \
                                                                                (_x8 ** c2) * \
                                                                                (_x9 ** c1)).reshape(mmm, 1))))
                                                    tempCount += 1
                                                    # print(tempCount)
                                                    if tempCount == mmm:
                                                        break
                                            else:
                                                continue
                                            break
                                        else:
                                            continue
                                        break
                                    else:
                                        continue
                                    break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        if B is not None:
            A = np.hstack((A, B))
        if C is not None:
            A = np.hstack((A, C))
        if D is not None:
            A = np.hstack((A, D))
        Taylor = np.linalg.solve(A, self.b)  # 求解k+2元一次方程
        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])  # 在开头加上0阶导
        return Taylor.tolist()[:taylorNum]  # 只保留0阶导+k阶导所有系数

    def _getData_15var(self, k=5, taylorNum=5005):  # 求解mmm元1次方程组
        # 展开至k阶k=mmm，共有Subject = m*(m+3)/2项，若要组成统一的矩阵，则需要再按5元最高阶的项数进行补0，先默认按5元展开至5阶最高阶一共128项
        n = 8  # 6元变量
        m = 10000  # m再大也不影响运行！！实际不需要展开至1000阶，但是展开至m阶的未知系数要比mmm大
        mmm = self.X.shape[0] - 1  # 除展开点有多少个样本点就全用上
        print('Matrix size', mmm)

        A = None
        B = None
        C = None
        D = None
        E = None
        _x0 = self._X[0][:-1] - self.expantionPoint[0]
        _x1 = self._X[1][:-1] - self.expantionPoint[1]
        _x2 = self._X[2][:-1] - self.expantionPoint[2]
        _x3 = self._X[3][:-1] - self.expantionPoint[3]
        _x4 = self._X[4][:-1] - self.expantionPoint[4]
        _x5 = self._X[5][:-1] - self.expantionPoint[5]
        _x6 = self._X[6][:-1] - self.expantionPoint[6]
        _x7 = self._X[7][:-1] - self.expantionPoint[7]
        _x8 = self._X[8][:-1] - self.expantionPoint[8]
        _x9 = self._X[8][:-1] - self.expantionPoint[9]
        _x10 = self._X[10][:-1] - self.expantionPoint[10]
        _x11 = self._X[11][:-1] - self.expantionPoint[11]
        _x12 = self._X[12][:-1] - self.expantionPoint[12]
        _x13 = self._X[13][:-1] - self.expantionPoint[13]
        _x14 = self._X[14][:-1] - self.expantionPoint[14]

        tempCount = 0
        for k in range(1, m + 1):
            for c1 in range(0, k + 1):
                for c2 in range(0, k + 1):
                    for c3 in range(0, k + 1):
                        for c4 in range(0, k + 1):
                            for c5 in range(0, k + 1):
                                for c6 in range(0, k + 1):
                                    for c7 in range(0, k + 1):
                                        for c8 in range(0, k + 1):
                                            for c9 in range(0, k + 1):
                                                for c10 in range(0, k + 1):
                                                    for c11 in range(0, k + 1):
                                                        for c12 in range(0, k + 1):
                                                            for c13 in range(0, k + 1):
                                                                for c14 in range(0, k + 1):
                                                                    if c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 <= k:
                                                                        c15 = k - c1 - c2 - c3 - c4 - c5 - c6 - c7 - c8 - c9 - c10 - c11 - c12 - c13 - c14
                                                                        if tempCount < 2000:
                                                                            if A is None:
                                                                                A = ((_x0 ** c15) * \
                                                                                     (_x1 ** c14) * \
                                                                                     (_x2 ** c13) * \
                                                                                     (_x3 ** c12) * \
                                                                                     (_x4 ** c11) * \
                                                                                     (_x5 ** c10) * \
                                                                                     (_x6 ** c9) * \
                                                                                     (_x7 ** c8) * \
                                                                                     (_x8 ** c7) * \
                                                                                     (_x9 ** c6) * \
                                                                                     (_x10 ** c5) * \
                                                                                     (_x11 ** c4) * \
                                                                                     (_x12 ** c3) * \
                                                                                     (_x13 ** c2) * \
                                                                                     (_x14 ** c1)).reshape(mmm, 1)
                                                                            else:
                                                                                A = np.hstack((A, (((_x0 ** c15) * \
                                                                                                    (_x1 ** c14) * \
                                                                                                    (_x2 ** c13) * \
                                                                                                    (_x3 ** c12) * \
                                                                                                    (_x4 ** c11) * \
                                                                                                    (_x5 ** c10) * \
                                                                                                    (_x6 ** c9) * \
                                                                                                    (_x7 ** c8) * \
                                                                                                    (_x8 ** c7) * \
                                                                                                    (_x9 ** c6) * \
                                                                                                    (_x10 ** c5) * \
                                                                                                    (_x11 ** c4) * \
                                                                                                    (_x12 ** c3) * \
                                                                                                    (_x13 ** c2) * \
                                                                                                    (
                                                                                                            _x14 ** c1)).reshape(
                                                                                    mmm, 1))))
                                                                        elif tempCount < 4000:
                                                                            if B is None:
                                                                                B = ((_x0 ** c15) * \
                                                                                     (_x1 ** c14) * \
                                                                                     (_x2 ** c13) * \
                                                                                     (_x3 ** c12) * \
                                                                                     (_x4 ** c11) * \
                                                                                     (_x5 ** c10) * \
                                                                                     (_x6 ** c9) * \
                                                                                     (_x7 ** c8) * \
                                                                                     (_x8 ** c7) * \
                                                                                     (_x9 ** c6) * \
                                                                                     (_x10 ** c5) * \
                                                                                     (_x11 ** c4) * \
                                                                                     (_x12 ** c3) * \
                                                                                     (_x13 ** c2) * \
                                                                                     (_x14 ** c1)).reshape(mmm, 1)
                                                                            else:
                                                                                B = np.hstack((B, (((_x0 ** c15) * \
                                                                                                    (_x1 ** c14) * \
                                                                                                    (_x2 ** c13) * \
                                                                                                    (_x3 ** c12) * \
                                                                                                    (_x4 ** c11) * \
                                                                                                    (_x5 ** c10) * \
                                                                                                    (_x6 ** c9) * \
                                                                                                    (_x7 ** c8) * \
                                                                                                    (_x8 ** c7) * \
                                                                                                    (_x9 ** c6) * \
                                                                                                    (_x10 ** c5) * \
                                                                                                    (_x11 ** c4) * \
                                                                                                    (_x12 ** c3) * \
                                                                                                    (_x13 ** c2) * \
                                                                                                    (
                                                                                                            _x14 ** c1)).reshape(
                                                                                    mmm, 1))))
                                                                        elif tempCount < 6000:
                                                                            if C is None:
                                                                                C = ((_x0 ** c15) * \
                                                                                     (_x1 ** c14) * \
                                                                                     (_x2 ** c13) * \
                                                                                     (_x3 ** c12) * \
                                                                                     (_x4 ** c11) * \
                                                                                     (_x5 ** c10) * \
                                                                                     (_x6 ** c9) * \
                                                                                     (_x7 ** c8) * \
                                                                                     (_x8 ** c7) * \
                                                                                     (_x9 ** c6) * \
                                                                                     (_x10 ** c5) * \
                                                                                     (_x11 ** c4) * \
                                                                                     (_x12 ** c3) * \
                                                                                     (_x13 ** c2) * \
                                                                                     (_x14 ** c1)).reshape(mmm, 1)
                                                                            else:
                                                                                C = np.hstack((C, (((_x0 ** c15) * \
                                                                                                    (_x1 ** c14) * \
                                                                                                    (_x2 ** c13) * \
                                                                                                    (_x3 ** c12) * \
                                                                                                    (_x4 ** c11) * \
                                                                                                    (_x5 ** c10) * \
                                                                                                    (_x6 ** c9) * \
                                                                                                    (_x7 ** c8) * \
                                                                                                    (_x8 ** c7) * \
                                                                                                    (_x9 ** c6) * \
                                                                                                    (_x10 ** c5) * \
                                                                                                    (_x11 ** c4) * \
                                                                                                    (_x12 ** c3) * \
                                                                                                    (_x13 ** c2) * \
                                                                                                    (
                                                                                                            _x14 ** c1)).reshape(
                                                                                    mmm, 1))))
                                                                        else:
                                                                            if D is None:
                                                                                D = ((_x0 ** c15) * \
                                                                                     (_x1 ** c14) * \
                                                                                     (_x2 ** c13) * \
                                                                                     (_x3 ** c12) * \
                                                                                     (_x4 ** c11) * \
                                                                                     (_x5 ** c10) * \
                                                                                     (_x6 ** c9) * \
                                                                                     (_x7 ** c8) * \
                                                                                     (_x8 ** c7) * \
                                                                                     (_x9 ** c6) * \
                                                                                     (_x10 ** c5) * \
                                                                                     (_x11 ** c4) * \
                                                                                     (_x12 ** c3) * \
                                                                                     (_x13 ** c2) * \
                                                                                     (_x14 ** c1)).reshape(mmm, 1)
                                                                            else:
                                                                                D = np.hstack((D, (((_x0 ** c15) * \
                                                                                                    (_x1 ** c14) * \
                                                                                                    (_x2 ** c13) * \
                                                                                                    (_x3 ** c12) * \
                                                                                                    (_x4 ** c11) * \
                                                                                                    (_x5 ** c10) * \
                                                                                                    (_x6 ** c9) * \
                                                                                                    (_x7 ** c8) * \
                                                                                                    (_x8 ** c7) * \
                                                                                                    (_x9 ** c6) * \
                                                                                                    (_x10 ** c5) * \
                                                                                                    (_x11 ** c4) * \
                                                                                                    (_x12 ** c3) * \
                                                                                                    (_x13 ** c2) * \
                                                                                                    (
                                                                                                            _x14 ** c1)).reshape(
                                                                                    mmm, 1))))
                                                                        tempCount += 1
                                                                        if tempCount == mmm:
                                                                            break
                                                                else:
                                                                    continue
                                                                break
                                                            else:
                                                                continue
                                                            break
                                                        else:
                                                            continue
                                                        break
                                                    else:
                                                        continue
                                                    break
                                                else:
                                                    continue
                                                break
                                            else:
                                                continue
                                            break
                                        else:
                                            continue
                                        break
                                    else:
                                        continue
                                    break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        if B is not None:
            A = np.hstack((A, B))
        if C is not None:
            A = np.hstack((A, C))
        if D is not None:
            A = np.hstack((A, D))
        Taylor = np.linalg.solve(A, self.b)
        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])
        return Taylor.tolist()[:taylorNum]

    def _getData_21var(self, k=5, taylorNum=5005):
        n = 8
        m = 10000
        mmm = self.X.shape[0] - 1
        print('Matrix size', mmm)

        A = None
        B = None
        C = None
        D = None
        E = None
        _x0 = self._X[0][:-1] - self.expantionPoint[0]
        _x1 = self._X[1][:-1] - self.expantionPoint[1]
        _x2 = self._X[2][:-1] - self.expantionPoint[2]
        _x3 = self._X[3][:-1] - self.expantionPoint[3]
        _x4 = self._X[4][:-1] - self.expantionPoint[4]
        _x5 = self._X[5][:-1] - self.expantionPoint[5]
        _x6 = self._X[6][:-1] - self.expantionPoint[6]
        _x7 = self._X[7][:-1] - self.expantionPoint[7]
        _x8 = self._X[8][:-1] - self.expantionPoint[8]
        _x9 = self._X[8][:-1] - self.expantionPoint[9]
        _x10 = self._X[10][:-1] - self.expantionPoint[10]
        _x11 = self._X[11][:-1] - self.expantionPoint[11]
        _x12 = self._X[12][:-1] - self.expantionPoint[12]
        _x13 = self._X[13][:-1] - self.expantionPoint[13]
        _x14 = self._X[14][:-1] - self.expantionPoint[14]
        _x15 = self._X[15][:-1] - self.expantionPoint[15]
        _x16 = self._X[16][:-1] - self.expantionPoint[16]
        _x17 = self._X[17][:-1] - self.expantionPoint[17]
        _x18 = self._X[18][:-1] - self.expantionPoint[18]
        _x19 = self._X[19][:-1] - self.expantionPoint[19]
        _x20 = self._X[20][:-1] - self.expantionPoint[20]

        tempCount = 0
        for k in range(1, m + 1):
            for c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20 in \
                    itertools.product(range(0, k + 1), range(0, k + 1), range(0, k + 1), range(0, k + 1),
                                      range(0, k + 1), range(0, k + 1), range(0, k + 1), range(0, k + 1),
                                      range(0, k + 1), range(0, k + 1), range(0, k + 1), range(0, k + 1),
                                      range(0, k + 1), range(0, k + 1), range(0, k + 1), range(0, k + 1),
                                      range(0, k + 1), range(0, k + 1), range(0, k + 1), range(0, k + 1)):
                if c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15 + c16 + c17 + c18 + c19 + c20 <= k:
                    c21 = k - c1 - c2 - c3 - c4 - c5 - c6 - c7 - c8 - c9 - c10 - c11 - c12 - c13 - c14 - c15 - c16 - c17 - c18 - c19 - c20
                    if tempCount < 2000:
                        if A is None:
                            A = ((_x0 ** c21) * \
                                 (_x1 ** c20) * \
                                 (_x2 ** c19) * \
                                 (_x3 ** c18) * \
                                 (_x4 ** c17) * \
                                 (_x5 ** c16) * \
                                 (_x6 ** c15) * \
                                 (_x7 ** c14) * \
                                 (_x8 ** c13) * \
                                 (_x9 ** c12) * \
                                 (_x10 ** c11) * \
                                 (_x11 ** c10) * \
                                 (_x12 ** c9) * \
                                 (_x13 ** c8) * \
                                 (_x14 ** c7) * \
                                 (_x15 ** c6) * \
                                 (_x16 ** c5) * \
                                 (_x17 ** c4) * \
                                 (_x18 ** c3) * \
                                 (_x19 ** c2) * \
                                 (_x20 ** c1)).reshape(mmm, 1)
                        else:
                            A = np.hstack((A, (((_x0 ** c21) * \
                                                (_x1 ** c20) * \
                                                (_x2 ** c19) * \
                                                (_x3 ** c18) * \
                                                (_x4 ** c17) * \
                                                (_x5 ** c16) * \
                                                (_x6 ** c15) * \
                                                (_x7 ** c14) * \
                                                (_x8 ** c13) * \
                                                (_x9 ** c12) * \
                                                (_x10 ** c11) * \
                                                (_x11 ** c10) * \
                                                (_x12 ** c9) * \
                                                (_x13 ** c8) * \
                                                (_x14 ** c7) * \
                                                (_x15 ** c6) * \
                                                (_x16 ** c5) * \
                                                (_x17 ** c4) * \
                                                (_x18 ** c3) * \
                                                (_x19 ** c2) * \
                                                (_x20 ** c1)).reshape(mmm, 1))))
                    elif tempCount < 4000:
                        if B is None:
                            B = ((_x0 ** c21) * \
                                 (_x1 ** c20) * \
                                 (_x2 ** c19) * \
                                 (_x3 ** c18) * \
                                 (_x4 ** c17) * \
                                 (_x5 ** c16) * \
                                 (_x6 ** c15) * \
                                 (_x7 ** c14) * \
                                 (_x8 ** c13) * \
                                 (_x9 ** c12) * \
                                 (_x10 ** c11) * \
                                 (_x11 ** c10) * \
                                 (_x12 ** c9) * \
                                 (_x13 ** c8) * \
                                 (_x14 ** c7) * \
                                 (_x15 ** c6) * \
                                 (_x16 ** c5) * \
                                 (_x17 ** c4) * \
                                 (_x18 ** c3) * \
                                 (_x19 ** c2) * \
                                 (_x20 ** c1)).reshape(mmm, 1)
                        else:
                            B = np.hstack((B, (((_x0 ** c21) * \
                                                (_x1 ** c20) * \
                                                (_x2 ** c19) * \
                                                (_x3 ** c18) * \
                                                (_x4 ** c17) * \
                                                (_x5 ** c16) * \
                                                (_x6 ** c15) * \
                                                (_x7 ** c14) * \
                                                (_x8 ** c13) * \
                                                (_x9 ** c12) * \
                                                (_x10 ** c11) * \
                                                (_x11 ** c10) * \
                                                (_x12 ** c9) * \
                                                (_x13 ** c8) * \
                                                (_x14 ** c7) * \
                                                (_x15 ** c6) * \
                                                (_x16 ** c5) * \
                                                (_x17 ** c4) * \
                                                (_x18 ** c3) * \
                                                (_x19 ** c2) * \
                                                (_x20 ** c1)).reshape(mmm, 1))))
                    elif tempCount < 6000:
                        if C is None:
                            C = ((_x0 ** c21) * \
                                 (_x1 ** c20) * \
                                 (_x2 ** c19) * \
                                 (_x3 ** c18) * \
                                 (_x4 ** c17) * \
                                 (_x5 ** c16) * \
                                 (_x6 ** c15) * \
                                 (_x7 ** c14) * \
                                 (_x8 ** c13) * \
                                 (_x9 ** c12) * \
                                 (_x10 ** c11) * \
                                 (_x11 ** c10) * \
                                 (_x12 ** c9) * \
                                 (_x13 ** c8) * \
                                 (_x14 ** c7) * \
                                 (_x15 ** c6) * \
                                 (_x16 ** c5) * \
                                 (_x17 ** c4) * \
                                 (_x18 ** c3) * \
                                 (_x19 ** c2) * \
                                 (_x20 ** c1)).reshape(mmm, 1)
                        else:
                            D = np.hstack((D, (((_x0 ** c21) * \
                                                (_x1 ** c20) * \
                                                (_x2 ** c19) * \
                                                (_x3 ** c18) * \
                                                (_x4 ** c17) * \
                                                (_x5 ** c16) * \
                                                (_x6 ** c15) * \
                                                (_x7 ** c14) * \
                                                (_x8 ** c13) * \
                                                (_x9 ** c12) * \
                                                (_x10 ** c11) * \
                                                (_x11 ** c10) * \
                                                (_x12 ** c9) * \
                                                (_x13 ** c8) * \
                                                (_x14 ** c7) * \
                                                (_x15 ** c6) * \
                                                (_x16 ** c5) * \
                                                (_x17 ** c4) * \
                                                (_x18 ** c3) * \
                                                (_x19 ** c2) * \
                                                (_x20 ** c1)).reshape(mmm, 1))))
                    else:
                        if D is None:
                            D = ((_x0 ** c21) * \
                                 (_x1 ** c20) * \
                                 (_x2 ** c19) * \
                                 (_x3 ** c18) * \
                                 (_x4 ** c17) * \
                                 (_x5 ** c16) * \
                                 (_x6 ** c15) * \
                                 (_x7 ** c14) * \
                                 (_x8 ** c13) * \
                                 (_x9 ** c12) * \
                                 (_x10 ** c11) * \
                                 (_x11 ** c10) * \
                                 (_x12 ** c9) * \
                                 (_x13 ** c8) * \
                                 (_x14 ** c7) * \
                                 (_x15 ** c6) * \
                                 (_x16 ** c5) * \
                                 (_x17 ** c4) * \
                                 (_x18 ** c3) * \
                                 (_x19 ** c2) * \
                                 (_x20 ** c1)).reshape(mmm, 1)
                        else:
                            D = np.hstack((D, (((_x0 ** c21) * \
                                                (_x1 ** c20) * \
                                                (_x2 ** c19) * \
                                                (_x3 ** c18) * \
                                                (_x4 ** c17) * \
                                                (_x5 ** c16) * \
                                                (_x6 ** c15) * \
                                                (_x7 ** c14) * \
                                                (_x8 ** c13) * \
                                                (_x9 ** c12) * \
                                                (_x10 ** c11) * \
                                                (_x11 ** c10) * \
                                                (_x12 ** c9) * \
                                                (_x13 ** c8) * \
                                                (_x14 ** c7) * \
                                                (_x15 ** c6) * \
                                                (_x16 ** c5) * \
                                                (_x17 ** c4) * \
                                                (_x18 ** c3) * \
                                                (_x19 ** c2) * \
                                                (_x20 ** c1)).reshape(mmm, 1))))
                    tempCount += 1
                    if tempCount == mmm:
                        break
            else:
                continue
            break
        print('Successfully generated matrix')
        if B is not None:
            A = np.hstack((A, B))
        if C is not None:
            A = np.hstack((A, C))
        if D is not None:
            A = np.hstack((A, D))
        Taylor = np.linalg.solve(A, self.b)
        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])
        return Taylor.tolist()[:taylorNum]

    def judge_Low_polynomial(self, lowLine=7, varNum=1):
        if self.low_nmse > 1e-5:
            return False
        return True

    def _cal_f_taylor_lowtaylor(self, k, taylor_log_flag=False):
        varNum = self.varNum
        if taylor_log_flag:
            Taylor = self.Taylor_log
        else:
            Taylor = self.taylor
        if varNum == 1:
            f = str(Taylor[0])
            for i in range(1, k):
                if Taylor[i] > 0:
                    f += '+' + str(Taylor[i]) + '*' + '(x0-' + str(self.expantionPointa0) + ')**' + str(i)
                elif Taylor[i] < 0:
                    f += str(Taylor[i]) + '*' + '(x0-' + str(self.expantionPointa0) + ')**' + str(i)
            f_taylor = sympify(f)
            f_taylor = f_taylor.expand()
            f_split = str(f_taylor).split()
            if taylor_log_flag == False:
                try:
                    self.bias = float(f_split[-2] + f_split[-1])
                except BaseException:
                    self.bias = 0.
            return f_taylor, sympify(str(f_taylor).split('*x0**7')[-1])
        else:
            f = str(Taylor[0])
            taylorNum = len(Taylor)
            tempCount = 1
            flag = true
            if varNum == 2:
                for i in range(1, k + 1):
                    if flag == true:
                        for j in range(0, i + 1):
                            if Taylor[tempCount] > 0:
                                f += '+' + str(Taylor[tempCount]) + \
                                     '*' + '(x0-' + str(self.expantionPoint[0]) + ')**' + str(i - j) + \
                                     '*' + '(x1-' + str(self.expantionPoint[1]) + ')**' + str(j)
                            elif Taylor[tempCount] < 0:
                                f += str(Taylor[tempCount]) + \
                                     '*' + '(x0-' + str(self.expantionPoint[0]) + ')**' + str(i - j) + \
                                     '*' + '(x1-' + str(self.expantionPoint[1]) + ')**' + str(j)
                            tempCount += 1
                            if tempCount == taylorNum:
                                flag = false
                                break
            elif varNum == 3:
                for m in range(1, k + 1):
                    if flag == true:
                        for c1 in range(0, m + 1):
                            if flag == true:
                                for c2 in range(0, m + 1):
                                    if c1 + c2 <= m:
                                        c3 = m - c1 - c2
                                        if Taylor[tempCount] > 0:
                                            f += '+' + str(Taylor[tempCount]) + \
                                                 '*' + '(x0-' + str(self.expantionPoint[0]) + ')**' + str(c3) + \
                                                 '*' + '(x1-' + str(self.expantionPoint[1]) + ')**' + str(c2) + \
                                                 '*' + '(x2-' + str(self.expantionPoint[2]) + ')**' + str(c1)
                                        elif Taylor[tempCount] < 0:
                                            f += str(Taylor[tempCount]) + \
                                                 '*' + '(x0-' + str(self.expantionPoint[0]) + ')**' + str(c3) + \
                                                 '*' + '(x1-' + str(self.expantionPoint[1]) + ')**' + str(c2) + \
                                                 '*' + '(x2-' + str(self.expantionPoint[2]) + ')**' + str(c1)
                                        tempCount += 1
                                        if tempCount == taylorNum:
                                            flag = false
                                            break
            elif varNum == 4:
                for m in range(1, k + 1):
                    if flag == true:
                        for c1 in range(0, m + 1):
                            if flag == true:
                                for c2 in range(0, m + 1):
                                    if flag == true:
                                        for c3 in range(0, m + 1):
                                            if c1 + c2 + c3 <= m:
                                                c4 = m - c1 - c2 - c3
                                                if Taylor[tempCount] > 0:
                                                    f += '+' + str(Taylor[tempCount]) + \
                                                         '*' + '(x0-' + str(self.expantionPoint[0]) + ')**' + str(c4) + \
                                                         '*' + '(x1-' + str(self.expantionPoint[1]) + ')**' + str(c3) + \
                                                         '*' + '(x2-' + str(self.expantionPoint[2]) + ')**' + str(c2) + \
                                                         '*' + '(x3-' + str(self.expantionPoint[3]) + ')**' + str(c1)
                                                elif Taylor[tempCount] < 0:
                                                    f += str(Taylor[tempCount]) + \
                                                         '*' + '(x0-' + str(self.expantionPoint[0]) + ')**' + str(c4) + \
                                                         '*' + '(x1-' + str(self.expantionPoint[1]) + ')**' + str(c3) + \
                                                         '*' + '(x2-' + str(self.expantionPoint[2]) + ')**' + str(c2) + \
                                                         '*' + '(x3-' + str(self.expantionPoint[3]) + ')**' + str(c1)
                                                tempCount += 1
                                                if tempCount == taylorNum:
                                                    flag = false
                                                    break
            elif varNum == 5:
                for m in range(1, k + 1):
                    if flag == true:
                        for c1 in range(0, m + 1):
                            if flag == true:
                                for c2 in range(0, m + 1):
                                    if flag == true:
                                        for c3 in range(0, m + 1):
                                            if flag == true:
                                                for c4 in range(0, m + 1):
                                                    if c1 + c2 + c3 + c4 <= m:
                                                        c5 = m - c1 - c2 - c3 - c4
                                                        if Taylor[tempCount] > 0:
                                                            f += '+' + str(Taylor[tempCount]) + \
                                                                 '*' + '(x0-' + str(
                                                                self.expantionPoint[0]) + ')**' + str(c5) + \
                                                                 '*' + '(x1-' + str(
                                                                self.expantionPoint[1]) + ')**' + str(c4) + \
                                                                 '*' + '(x2-' + str(
                                                                self.expantionPoint[2]) + ')**' + str(c3) + \
                                                                 '*' + '(x3-' + str(
                                                                self.expantionPoint[3]) + ')**' + str(c2) + \
                                                                 '*' + '(x4-' + str(
                                                                self.expantionPoint[4]) + ')**' + str(c1)
                                                        elif Taylor[tempCount] < 0:
                                                            f += str(Taylor[tempCount]) + \
                                                                 '*' + '(x0-' + str(
                                                                self.expantionPoint[0]) + ')**' + str(c5) + \
                                                                 '*' + '(x1-' + str(
                                                                self.expantionPoint[1]) + ')**' + str(c4) + \
                                                                 '*' + '(x2-' + str(
                                                                self.expantionPoint[2]) + ')**' + str(c3) + \
                                                                 '*' + '(x3-' + str(
                                                                self.expantionPoint[3]) + ')**' + str(c2) + \
                                                                 '*' + '(x4-' + str(
                                                                self.expantionPoint[4]) + ')**' + str(c1)
            elif varNum == 6:
                for m in range(1, k + 1):
                    if flag == true:
                        for c1 in range(0, m + 1):
                            if flag == true:
                                for c2 in range(0, m + 1):
                                    if flag == true:
                                        for c3 in range(0, m + 1):
                                            if flag == true:
                                                for c4 in range(0, m + 1):
                                                    if flag == true:
                                                        for c5 in range(0, m + 1):
                                                            if c1 + c2 + c3 + c4 + c5 <= m:
                                                                c6 = m - c1 - c2 - c3 - c4 - c5
                                                                if Taylor[tempCount] > 0:
                                                                    f += '+' + str(Taylor[tempCount]) + \
                                                                         '*' + '(x0-' + str(
                                                                        self.expantionPoint[0]) + ')**' + str(c6) + \
                                                                         '*' + '(x1-' + str(
                                                                        self.expantionPoint[1]) + ')**' + str(c5) + \
                                                                         '*' + '(x2-' + str(
                                                                        self.expantionPoint[2]) + ')**' + str(c4) + \
                                                                         '*' + '(x3-' + str(
                                                                        self.expantionPoint[3]) + ')**' + str(c3) + \
                                                                         '*' + '(x4-' + str(
                                                                        self.expantionPoint[4]) + ')**' + str(c2) + \
                                                                         '*' + '(x5-' + str(
                                                                        self.expantionPoint[5]) + ')**' + str(c1)
                                                                elif Taylor[tempCount] < 0:
                                                                    f += str(Taylor[tempCount]) + \
                                                                         '*' + '(x0-' + str(
                                                                        self.expantionPoint[0]) + ')**' + str(c6) + \
                                                                         '*' + '(x1-' + str(
                                                                        self.expantionPoint[1]) + ')**' + str(c5) + \
                                                                         '*' + '(x2-' + str(
                                                                        self.expantionPoint[2]) + ')**' + str(c4) + \
                                                                         '*' + '(x3-' + str(
                                                                        self.expantionPoint[3]) + ')**' + str(c3) + \
                                                                         '*' + '(x4-' + str(
                                                                        self.expantionPoint[4]) + ')**' + str(c2) + \
                                                                         '*' + '(x5-' + str(
                                                                        self.expantionPoint[5]) + ')**' + str(c1)
            elif varNum == 8:
                for m in range(1, k + 1):
                    if flag == true:
                        for c1 in range(0, m + 1):
                            if flag == true:
                                for c2 in range(0, m + 1):
                                    if flag == true:
                                        for c3 in range(0, m + 1):
                                            if flag == true:
                                                for c4 in range(0, m + 1):
                                                    if flag == true:
                                                        for c5 in range(0, m + 1):
                                                            if flag == true:
                                                                for c6 in range(0, m + 1):
                                                                    if flag == true:
                                                                        for c7 in range(0, m + 1):
                                                                            if c1 + c2 + c3 + c4 + c5 + c6 + c7 <= m:
                                                                                c8 = m - c1 - c2 - c3 - c4 - c5 - c6 - c7
                                                                                if Taylor[tempCount] > 0:
                                                                                    f += '+' + str(Taylor[tempCount]) + \
                                                                                         '*' + '(x0-' + str(
                                                                                        self.expantionPoint[
                                                                                            0]) + ')**' + str(c8) + \
                                                                                         '*' + '(x1-' + str(
                                                                                        self.expantionPoint[
                                                                                            1]) + ')**' + str(c7) + \
                                                                                         '*' + '(x2-' + str(
                                                                                        self.expantionPoint[
                                                                                            2]) + ')**' + str(c6) + \
                                                                                         '*' + '(x3-' + str(
                                                                                        self.expantionPoint[
                                                                                            3]) + ')**' + str(c5) + \
                                                                                         '*' + '(x4-' + str(
                                                                                        self.expantionPoint[
                                                                                            4]) + ')**' + str(c4) + \
                                                                                         '*' + '(x5-' + str(
                                                                                        self.expantionPoint[
                                                                                            5]) + ')**' + str(c3) + \
                                                                                         '*' + '(x6-' + str(
                                                                                        self.expantionPoint[
                                                                                            6]) + ')**' + str(c2) + \
                                                                                         '*' + '(x7-' + str(
                                                                                        self.expantionPoint[
                                                                                            7]) + ')**' + str(c1)
                                                                                elif Taylor[tempCount] < 0:
                                                                                    f += str(Taylor[tempCount]) + \
                                                                                         '*' + '(x0-' + str(
                                                                                        self.expantionPoint[
                                                                                            0]) + ')**' + str(c8) + \
                                                                                         '*' + '(x1-' + str(
                                                                                        self.expantionPoint[
                                                                                            1]) + ')**' + str(c7) + \
                                                                                         '*' + '(x2-' + str(
                                                                                        self.expantionPoint[
                                                                                            2]) + ')**' + str(c6) + \
                                                                                         '*' + '(x3-' + str(
                                                                                        self.expantionPoint[
                                                                                            3]) + ')**' + str(c5) + \
                                                                                         '*' + '(x4-' + str(
                                                                                        self.expantionPoint[
                                                                                            4]) + ')**' + str(c4) + \
                                                                                         '*' + '(x5-' + str(
                                                                                        self.expantionPoint[
                                                                                            5]) + ')**' + str(c3) + \
                                                                                         '*' + '(x6-' + str(
                                                                                        self.expantionPoint[
                                                                                            6]) + ')**' + str(c2) + \
                                                                                         '*' + '(x7-' + str(
                                                                                        self.expantionPoint[
                                                                                            7]) + ')**' + str(c1)
            elif varNum == 9:
                for m in range(1, k + 1):
                    if flag == true:
                        for c1 in range(0, m + 1):
                            if flag == true:
                                for c2 in range(0, m + 1):
                                    if flag == true:
                                        for c3 in range(0, m + 1):
                                            if flag == true:
                                                for c4 in range(0, m + 1):
                                                    if flag == true:
                                                        for c5 in range(0, m + 1):
                                                            if flag == true:
                                                                for c6 in range(0, m + 1):
                                                                    if flag == true:
                                                                        for c7 in range(0, m + 1):
                                                                            if flag == true:
                                                                                for c8 in range(0, m + 1):
                                                                                    if c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 <= m:
                                                                                        c9 = m - c1 - c2 - c3 - c4 - c5 - c6 - c7 - c8
                                                                                        if Taylor[tempCount] > 0:
                                                                                            f += '+' + str(
                                                                                                Taylor[tempCount]) + \
                                                                                                 '*' + '(x0-' + str(
                                                                                                self.expantionPoint[
                                                                                                    0]) + ')**' + str(
                                                                                                c9) + \
                                                                                                 '*' + '(x1-' + str(
                                                                                                self.expantionPoint[
                                                                                                    1]) + ')**' + str(
                                                                                                c8) + \
                                                                                                 '*' + '(x2-' + str(
                                                                                                self.expantionPoint[
                                                                                                    2]) + ')**' + str(
                                                                                                c7) + \
                                                                                                 '*' + '(x3-' + str(
                                                                                                self.expantionPoint[
                                                                                                    3]) + ')**' + str(
                                                                                                c6) + \
                                                                                                 '*' + '(x4-' + str(
                                                                                                self.expantionPoint[
                                                                                                    4]) + ')**' + str(
                                                                                                c5) + \
                                                                                                 '*' + '(x5-' + str(
                                                                                                self.expantionPoint[
                                                                                                    5]) + ')**' + str(
                                                                                                c4) + \
                                                                                                 '*' + '(x6-' + str(
                                                                                                self.expantionPoint[
                                                                                                    6]) + ')**' + str(
                                                                                                c3) + \
                                                                                                 '*' + '(x7-' + str(
                                                                                                self.expantionPoint[
                                                                                                    7]) + ')**' + str(
                                                                                                c2) + \
                                                                                                 '*' + '(x8-' + str(
                                                                                                self.expantionPoint[
                                                                                                    8]) + ')**' + str(
                                                                                                c1)
                                                                                        elif Taylor[tempCount] < 0:
                                                                                            f += str(
                                                                                                Taylor[tempCount]) + \
                                                                                                 '*' + '(x0-' + str(
                                                                                                self.expantionPoint[
                                                                                                    0]) + ')**' + str(
                                                                                                c9) + \
                                                                                                 '*' + '(x1-' + str(
                                                                                                self.expantionPoint[
                                                                                                    1]) + ')**' + str(
                                                                                                c8) + \
                                                                                                 '*' + '(x2-' + str(
                                                                                                self.expantionPoint[
                                                                                                    2]) + ')**' + str(
                                                                                                c7) + \
                                                                                                 '*' + '(x3-' + str(
                                                                                                self.expantionPoint[
                                                                                                    3]) + ')**' + str(
                                                                                                c6) + \
                                                                                                 '*' + '(x4-' + str(
                                                                                                self.expantionPoint[
                                                                                                    4]) + ')**' + str(
                                                                                                c5) + \
                                                                                                 '*' + '(x5-' + str(
                                                                                                self.expantionPoint[
                                                                                                    5]) + ')**' + str(
                                                                                                c4) + \
                                                                                                 '*' + '(x6-' + str(
                                                                                                self.expantionPoint[
                                                                                                    6]) + ')**' + str(
                                                                                                c3) + \
                                                                                                 '*' + '(x7-' + str(
                                                                                                self.expantionPoint[
                                                                                                    7]) + ')**' + str(
                                                                                                c2) + \
                                                                                                 '*' + '(x8-' + str(
                                                                                                self.expantionPoint[
                                                                                                    8]) + ')**' + str(
                                                                                                c1)
            elif varNum == 10:
                for m in range(1, k + 1):
                    for c1 in range(0, m + 1):
                        for c2 in range(0, m + 1):
                            for c3 in range(0, m + 1):
                                for c4 in range(0, m + 1):
                                    for c5 in range(0, m + 1):
                                        for c6 in range(0, m + 1):
                                            for c7 in range(0, m + 1):
                                                for c8 in range(0, m + 1):
                                                    for c9 in range(0, m + 1):
                                                        if c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 <= m:
                                                            c10 = m - c1 - c2 - c3 - c4 - c5 - c6 - c7 - c8 - c9
                                                            if Taylor[tempCount] > 0:
                                                                f += '+' + str(Taylor[tempCount]) + \
                                                                     '*' + '(x0-' + str(
                                                                    self.expantionPoint[0]) + ')**' + str(c9) + \
                                                                     '*' + '(x1-' + str(
                                                                    self.expantionPoint[1]) + ')**' + str(c8) + \
                                                                     '*' + '(x2-' + str(
                                                                    self.expantionPoint[2]) + ')**' + str(c7) + \
                                                                     '*' + '(x3-' + str(
                                                                    self.expantionPoint[3]) + ')**' + str(c6) + \
                                                                     '*' + '(x4-' + str(
                                                                    self.expantionPoint[4]) + ')**' + str(c5) + \
                                                                     '*' + '(x5-' + str(
                                                                    self.expantionPoint[5]) + ')**' + str(c4) + \
                                                                     '*' + '(x6-' + str(
                                                                    self.expantionPoint[6]) + ')**' + str(c3) + \
                                                                     '*' + '(x7-' + str(
                                                                    self.expantionPoint[7]) + ')**' + str(c2) + \
                                                                     '*' + '(x8-' + str(
                                                                    self.expantionPoint[8]) + ')**' + str(c1) + \
                                                                     '*' + '(x9-' + str(
                                                                    self.expantionPoint[9]) + ')**' + str(c1)
                                                            elif Taylor[tempCount] < 0:
                                                                f += str(Taylor[tempCount]) + \
                                                                     '*' + '(x0-' + str(
                                                                    self.expantionPoint[0]) + ')**' + str(c9) + \
                                                                     '*' + '(x1-' + str(
                                                                    self.expantionPoint[1]) + ')**' + str(c8) + \
                                                                     '*' + '(x2-' + str(
                                                                    self.expantionPoint[2]) + ')**' + str(c7) + \
                                                                     '*' + '(x3-' + str(
                                                                    self.expantionPoint[3]) + ')**' + str(c6) + \
                                                                     '*' + '(x4-' + str(
                                                                    self.expantionPoint[4]) + ')**' + str(c5) + \
                                                                     '*' + '(x5-' + str(
                                                                    self.expantionPoint[5]) + ')**' + str(c4) + \
                                                                     '*' + '(x6-' + str(
                                                                    self.expantionPoint[6]) + ')**' + str(c3) + \
                                                                     '*' + '(x7-' + str(
                                                                    self.expantionPoint[7]) + ')**' + str(c2) + \
                                                                     '*' + '(x8-' + str(
                                                                    self.expantionPoint[8]) + ')**' + str(c1) + \
                                                                     '*' + '(x9-' + str(
                                                                    self.expantionPoint[9]) + ')**' + str(c1)
            elif varNum == 15:
                for m in range(1, k + 1):
                    for c1 in range(0, m + 1):
                        for c2 in range(0, m + 1):
                            for c3 in range(0, m + 1):
                                for c4 in range(0, m + 1):
                                    for c5 in range(0, m + 1):
                                        for c6 in range(0, m + 1):
                                            for c7 in range(0, m + 1):
                                                for c8 in range(0, m + 1):
                                                    for c9 in range(0, m + 1):
                                                        for c10 in range(0, m + 1):
                                                            for c11 in range(0, m + 1):
                                                                for c12 in range(0, m + 1):
                                                                    for c13 in range(0, m + 1):
                                                                        for c14 in range(0, m + 1):
                                                                            if c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 <= m:
                                                                                c15 = m - c1 - c2 - c3 - c4 - c5 - c6 - c7 - c8 - c9 - c10 - c11 - c12 - c13 - c14
                                                                                if Taylor[tempCount] > 0:
                                                                                    f += '+' + str(Taylor[tempCount]) + \
                                                                                         '*' + '(x0-' + str(
                                                                                        self.expantionPoint[
                                                                                            0]) + ')**' + str(c15) + \
                                                                                         '*' + '(x1-' + str(
                                                                                        self.expantionPoint[
                                                                                            1]) + ')**' + str(c14) + \
                                                                                         '*' + '(x2-' + str(
                                                                                        self.expantionPoint[
                                                                                            2]) + ')**' + str(c13) + \
                                                                                         '*' + '(x3-' + str(
                                                                                        self.expantionPoint[
                                                                                            3]) + ')**' + str(c12) + \
                                                                                         '*' + '(x4-' + str(
                                                                                        self.expantionPoint[
                                                                                            4]) + ')**' + str(c11) + \
                                                                                         '*' + '(x5-' + str(
                                                                                        self.expantionPoint[
                                                                                            5]) + ')**' + str(c10) + \
                                                                                         '*' + '(x6-' + str(
                                                                                        self.expantionPoint[
                                                                                            6]) + ')**' + str(c9) + \
                                                                                         '*' + '(x7-' + str(
                                                                                        self.expantionPoint[
                                                                                            7]) + ')**' + str(c8) + \
                                                                                         '*' + '(x8-' + str(
                                                                                        self.expantionPoint[
                                                                                            8]) + ')**' + str(c7) + \
                                                                                         '*' + '(x9-' + str(
                                                                                        self.expantionPoint[
                                                                                            9]) + ')**' + str(c6) + \
                                                                                         '*' + '(x10-' + str(
                                                                                        self.expantionPoint[
                                                                                            10]) + ')**' + str(c5) + \
                                                                                         '*' + '(x11-' + str(
                                                                                        self.expantionPoint[
                                                                                            11]) + ')**' + str(c4) + \
                                                                                         '*' + '(x12-' + str(
                                                                                        self.expantionPoint[
                                                                                            12]) + ')**' + str(c3) + \
                                                                                         '*' + '(x13-' + str(
                                                                                        self.expantionPoint[
                                                                                            13]) + ')**' + str(c2) + \
                                                                                         '*' + '(x14-' + str(
                                                                                        self.expantionPoint[
                                                                                            14]) + ')**' + str(c1)
                                                                                elif Taylor[tempCount] < 0:
                                                                                    f += str(Taylor[tempCount]) + \
                                                                                         '*' + '(x0-' + str(
                                                                                        self.expantionPoint[
                                                                                            0]) + ')**' + str(c15) + \
                                                                                         '*' + '(x1-' + str(
                                                                                        self.expantionPoint[
                                                                                            1]) + ')**' + str(c14) + \
                                                                                         '*' + '(x2-' + str(
                                                                                        self.expantionPoint[
                                                                                            2]) + ')**' + str(c13) + \
                                                                                         '*' + '(x3-' + str(
                                                                                        self.expantionPoint[
                                                                                            3]) + ')**' + str(c12) + \
                                                                                         '*' + '(x4-' + str(
                                                                                        self.expantionPoint[
                                                                                            4]) + ')**' + str(c11) + \
                                                                                         '*' + '(x5-' + str(
                                                                                        self.expantionPoint[
                                                                                            5]) + ')**' + str(c10) + \
                                                                                         '*' + '(x6-' + str(
                                                                                        self.expantionPoint[
                                                                                            6]) + ')**' + str(c9) + \
                                                                                         '*' + '(x7-' + str(
                                                                                        self.expantionPoint[
                                                                                            7]) + ')**' + str(c8) + \
                                                                                         '*' + '(x8-' + str(
                                                                                        self.expantionPoint[
                                                                                            8]) + ')**' + str(c7) + \
                                                                                         '*' + '(x9-' + str(
                                                                                        self.expantionPoint[
                                                                                            9]) + ')**' + str(c6) + \
                                                                                         '*' + '(x10-' + str(
                                                                                        self.expantionPoint[
                                                                                            10]) + ')**' + str(c5) + \
                                                                                         '*' + '(x11-' + str(
                                                                                        self.expantionPoint[
                                                                                            11]) + ')**' + str(c4) + \
                                                                                         '*' + '(x12-' + str(
                                                                                        self.expantionPoint[
                                                                                            12]) + ')**' + str(c3) + \
                                                                                         '*' + '(x13-' + str(
                                                                                        self.expantionPoint[
                                                                                            13]) + ')**' + str(c2) + \
                                                                                         '*' + '(x14-' + str(
                                                                                        self.expantionPoint[
                                                                                            14]) + ')**' + str(c1)

            f_taylor = sympify(f)
            f_taylor = f_taylor.expand()
            f_split = str(f_taylor).split()
            if taylor_log_flag == False:
                try:
                    self.bias = float(f_split[-2] + f_split[-1])
                except BaseException:
                    self.bias = 0.
            return f_taylor

    def _getTaylorPolynomial(self, varNum=1):
        Taylor = self.taylor
        if varNum == 1:
            self.f_taylor, self.f_low_taylor = self._cal_f_taylor_lowtaylor(k=14)
            y_pred = self._calY(self.f_taylor)
            y_low_pred = self._calY(self.f_low_taylor)
            nmse = mean_squared_error(self.Y, y_pred)
            low_nmse = mean_squared_error(self.Y, y_low_pred)
            print('NMSE of Taylor polynomal：', nmse)
            print('NMSE of Low order Taylor polynomial：', low_nmse)
            self.nmse = nmse
            self.low_nmse = low_nmse
            return self.f_taylor
        else:
            count1 = 2
            count2 = 3
            if self.varNum == 2:
                count1 = 5
                count2 = 9
            elif self.varNum == 3:
                count1 = 4
                count2 = 8
            elif self.varNum in [4, 5, 6]:
                count1 = 3
                count2 = 7

            for k in range(1, count1):
                test_f_k = self._cal_f_taylor_lowtaylor(k)
                test_y_pred = self._calY(test_f_k)
                test_nmse = mean_squared_error(self.Y, test_y_pred)
                print('NMSE expanded to order k，k=', k, 'nmse=', test_nmse)
                if test_nmse < self.low_nmse:
                    self.low_nmse = test_nmse
                    self.f_low_taylor = test_f_k
            try:
                for k in range(count1, count2):
                    test_f_k = self._cal_f_taylor_lowtaylor(k)
                    test_y_pred = self._calY(test_f_k)
                    test_nmse = mean_squared_error(self.Y, test_y_pred)
                    print('NMSE expanded to order k，k=', k, 'nmse=', test_nmse)
                    if test_nmse < self.nmse:
                        self.nmse = test_nmse
                        self.f_taylor = test_f_k
            except BaseException:
                print('sympify error')
            try:
                self.f_taylor_log = self._cal_f_taylor_lowtaylor(k=8, taylor_log_flag=True)
            except BaseException:
                self.f_taylor_log = 0
                print('f_taylor_log error')
            y_pred = self._calY(self.f_taylor)
            y_low_pred = self._calY(self.f_low_taylor)
            nmse = mean_squared_error(self.Y, y_pred)
            low_nmse = mean_squared_error(self.Y, y_low_pred)
            print('NMSE of Taylor polynomal：', nmse)
            print('NMSE of Low order Taylor polynomial：', low_nmse)

            self.nmse = nmse
            self.low_nmse = low_nmse
            return self.f_taylor

    def _calY(self, f, _x=None, X=None):
        y_pred = []
        len1, len2 = 0, 0
        if _x is None:
            _x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21,
                  x22, x23, x24, x25, x26, x27, x28, x29]
        if X is None:
            X = self._X
            len2 = self.varNum
        else:
            len2 = len(X)
        len1 = X[0].shape[0]
        for i in range(len1):
            _sub = {}
            for j in range(len2):
                _sub.update({_x[j]: X[j][i]})
            y_pred.append(f.evalf(subs=_sub))
        return y_pred

    @timeout_decorator.timeout(10, use_signals=False)
    def cal_critical_point(self, fx, x):
        if self.varNum == 2:
            return solve([fx[0], fx[1]], [x[0], x[1]])

    def judge_Bound(self):
        if self.nihe_flag == False:
            y_bound, var_bound = [], []
            _X = copy.deepcopy(self._X)
            for i in range(len(_X)):
                _X[i].sort()
                var_bound.extend([_X[i][0], _X[i][-1]])
            _Y = self.Y.reshape(-1)
            _Y.sort()
            y_bound.extend([_Y[0], _Y[-1]])
            return [y_bound, var_bound]
        else:
            y_bound, var_bound = [], []
            _X = copy.deepcopy(self._X)
            for i in range(len(_X)):
                _X[i].sort()
                var_bound.extend([_X[i][0], _X[i][-1]])
            Y = copy.deepcopy(self.Y)
            Y.sort()
            y_bound.extend([Y[0], Y[-1]])

            f_diff = []
            for i in range(len(self._X)):
                f_diff.append(sympify(diff(self.f_taylor, self._x[i])))
            try:
                critical_point = self.cal_critical_point(f_diff, self._x[:len(self._X)])
            except BaseException:
                critical_point = None
            if critical_point is not None:
                for c in critical_point:
                    if 'I' not in str(c) and not any(
                            [c[0] < var_bound[[i][0]] and c[1] > var_bound[i][1] for i in range(len(c))]):
                        _sub = {}
                        for i in range(len(c)):
                            _sub.update({self._x[i]: c[i]})
                        y_bound.append(self.f_taylor.evalf(subs=_sub))
                        print('Critical Point', c)
            y_bound.sort()
            return [[y_bound[0], y_bound[-1]], var_bound]

    def judge_monotonicity(self, Num=1):
        Increase, Decrease = False, False
        X, Y = copy.deepcopy(self.X), copy.deepcopy(self.Y)
        Y_index = np.argsort(Y, axis=0)
        Y_index = Y_index.reshape(-1)
        for i in range(1, Y_index.shape[0]):
            Increase_flag = not any([(X[Y_index[i]][j] < X[Y_index[i - 1]][j]) for j in range(X.shape[1])])
            if Increase_flag:
                Increase = True
            Decrease_flag = not any([(X[Y_index[i]][j] > X[Y_index[i - 1]][j]) for j in range(X.shape[1])])
            if Decrease_flag:
                Decrease = True
        if Increase == True and Decrease == False:
            if Num == 1:
                print('Monotone increasing function！！！')
            else:
                print('concave function')
            return 1
        elif Increase == False and Decrease == True:
            if Num == 1:
                print('Monotone decreasing function！！！')
            else:
                print('convex function')
            return 2
        if Num == 1:
            print('Non increasing and non decreasing function！！！')
        else:
            print(' concavity and convexity')
        return -1

    def judge_program_monotonicity(self, Num=1):  # 适合任意一维和多维的情况
        Increase, Decrease = False, False
        f_ = diff(self.f_taylor, self.x, Num)  # 求f的Num阶导数
        for x_ in self.X0:
            if f_.evalf(subs={x: x_}) >= 0:
                Increase = True
            if f_.evalf(subs={x: x_}) <= 0:
                Decrease = True
        if Increase == True and Decrease == False:
            if Num == 1:
                print('Monotone increasing function！！！')
            else:
                print('concave function')
            return 1
        elif Increase == False and Decrease == True:
            if Num == 1:
                self.di_jian_flag = True
                print('Monotone decreasing function！！！')
            else:
                print('convex function')
            return 2
        if Num == 1:
            print('Non increasing non decreasing function！！！')
        else:
            print('no concavity and convexity')
        return -1

    def judge_concavityConvexity(self):
        return self.judge_monotonicity(Num=2)

    def cal_power_expr(self, expr):
        expr = expr.split('*')
        j = 0
        for i in range(len(expr)):
            if expr[j] == '':
                expr.pop(j)
            else:
                j += 1
        count = 0
        for i in range(1, len(expr) - 1):
            if 'x' in expr[i] and expr[i + 1].isdigit():
                count += int(expr[i + 1])
            elif 'x' in expr[i] and 'x' in expr[i + 1]:
                count += 1
        if 'x' in expr[-1]:
            count += 1
        return count

    def judge_parity(self):
        '''
                odd function：1
                even function：2
        '''

        if self.nihe_flag:
            if self.bias != 0:
                f_taylor = str(self.f_taylor).split()[:-2]
            else:
                f_taylor = str(self.f_taylor).split()
            Y = self.Y - self.bias
            f_odd, f_even = '', ''
            if self.cal_power_expr(f_taylor[0]) % 2 == 1:
                f_odd += f_taylor[0]
            else:
                f_even += f_taylor[0]
            for i in range(2, len(f_taylor), 2):
                if self.cal_power_expr(f_taylor[i - 1] + f_taylor[i]) % 2 == 1:
                    f_odd += f_taylor[i - 1] + f_taylor[i]
                else:
                    f_even += f_taylor[i - 1] + f_taylor[i]
            f_odd, f_even = sympify(f_odd), sympify(f_even)
            Jishu, Oushu, nmse_odd, nmse_even = False, False, 0, 0
            y_pred = self._calY(f_odd)
            nmse_odd = mean_squared_error(Y, y_pred)
            y_pred = self._calY(f_even)
            nmse_even = mean_squared_error(Y, y_pred)
            print('NMSE of parity function', nmse_odd, nmse_even, sep='\n')
            if nmse_odd < 0.01:
                Jishu = True
            if nmse_even < 0.01:
                Oushu = True
            if Jishu == True and Oushu == False:
                print('odd function！！！')
                self.parity_flag = True
                return 1
            elif Jishu == False and Oushu == True:
                self.parity_flag = True
                print('even function！！！')
                return 2
        print('non odd non even function！！！')
        return -1

    def judge_program_parity(self):
        Jishu, Oushu = False, False
        f = self.f_taylor
        for x_ in self.X0:
            if abs(f.evalf(subs={x: -1 * x_}) + f.evalf(subs={x: x_})) < 0.001:
                Jishu = True
            elif abs(f.evalf(subs={x: -1 * x_}) - f.evalf(subs={x: x_})) < 0.001:
                Oushu = True
            else:
                print('non odd non even function！！！')
                return -1
        if Jishu == True and Oushu == False:
            print('odd function！！！')
            return 1
        elif Jishu == False and Oushu == True:
            print('even function！！！')
            return 2

    def _cal_add_separability(self, multi_flag=False, expantionpoint=None):
        f_taylor = str(copy.deepcopy(self.f_taylor))
        for i in range(len(self._x_right)):
            f_taylor = f_taylor.replace(str(self._x_right[i]), str(self._mid_right[i])) + '-(' + str(self.bias) + ')'
        self.f_left_taylor = f_taylor
        f_taylor = str(copy.deepcopy(self.f_taylor))
        for i in range(len(self._x_left)):
            f_taylor = f_taylor.replace(str(self._x_left[i]), str(self._mid_left[i]))
        self.f_right_taylor = f_taylor

        self.f_left_taylor = sympify(self.f_left_taylor)
        self.f_right_taylor = sympify(self.f_right_taylor)
        Y_left = self._calY(self.f_left_taylor, self._x_left, self._X_left)
        Y_right = self._calY(self.f_right_taylor, self._x_right, self._X_right)
        if multi_flag:
            f_taylor = str(copy.deepcopy(self.f_taylor))
            for i in range(len(self._x)):
                f_taylor = f_taylor.replace(str(self._x[i]), str(expantionpoint[i]))
            self.f_mid = eval(f_taylor)
        len_Y = len(Y_left)
        Y_left, Y_right = np.array(Y_left), np.array(Y_right)
        self.Y_left, self.Y_right = Y_left.reshape(len_Y, 1), Y_right.reshape(len_Y, 1)
        return None

    def judge_additi_separability(self, f_taylor=None, taylor_log_flag=False):
        for i in range(len(self._x) - 1):
            _x = copy.deepcopy(self._x)
            _x_left = _x.pop(i)
            _x_right = _x
            if f_taylor == None:
                f_taylor = self.f_taylor
            f_taylor_split = str(f_taylor).split()
            f_temp = []
            for subF in f_taylor_split:
                if any([str(_x_left) in subF and str(_x_right[j]) in subF for j in range(len(_x_right))]):
                    f_temp = f_temp[:-1]
                else:
                    f_temp.append(subF)

            f = ''
            for temp in f_temp:
                f += temp
            f = sympify(f)
            y_pred = self._calY(f, self._x, self._X)
            if taylor_log_flag:
                nmse = mean_squared_error(self.Y_log, y_pred)
            else:
                nmse = mean_squared_error(self.Y, y_pred)
            if nmse > 0.001:
                continue
            else:
                print('addition separable')
                return True

    def judge_multi_separability(self):
        '''multiplicative separability discrimination'''
        _expantionpoint = copy.deepcopy(self.expantionPoint)
        if self.expantionPoint[-1] == 0:
            _expantionpoint = _expantionpoint + 0.2
        for i in range(len(self._x) - 1):
            f_taylor = self.f_taylor
            _x = copy.deepcopy(self._x)
            self._x_left = [_x[i]]
            _x.pop(i)
            self._x_right = _x
            expantionpoint = copy.deepcopy(_expantionpoint).tolist()
            self._mid_left = [expantionpoint[i]]
            expantionpoint.pop(i)
            self._mid_right = expantionpoint[:-1]
            _X = copy.deepcopy(self._X)
            self._X_left = [_X.pop(i)]
            try:
                a = _X[0].shape[0]
            except BaseException:
                _X = [_X]
            self._X_right = _X
            self._cal_add_separability(multi_flag=True, expantionpoint=expantionpoint)
            nmse = mean_squared_error(self.Y, self.Y_left * self.Y_right / self.f_mid)
            if nmse < 0.001:
                print('multiplication separable')
                print('multi_nmse = ', nmse)
                return True
            else:
                try:
                    return self.judge_additi_separability(f_taylor=self.f_taylor_log, taylor_log_flag=True)
                except BaseException:
                    return False

        return False

    def change_Y(self, Y):
        if Y is None:
            return None
        if self.parity_flag:
            if abs(self.bias) > 1e-5:
                Y -= self.bias
        if self.di_jian_flag:
            return Y * (-1)
        else:
            return Y


class Metrics2(Metrics):

    def __init__(self, f_taylor, _x, X, Y):
        self.f_taylor = f_taylor
        self.f_low_taylor = None
        self.x0, self.x1, self.x2, self.x3, self.x4 = symbols("x0,x1,x2,x3,x4")
        self._x = _x
        self.bias, self.low_nmse = 0., 0.
        self.varNum = X.shape[1]
        self.Y_left, self.Y_right, self.Y_right_temp = None, None, None
        self.X_left, self.X_right = None, None
        self.midpoint = None
        self.parity_flag = False
        self.di_jian_flag = False
        self.expantionPoint = np.append(copy.deepcopy(X[0]), Y[0][0])
        self.X = copy.deepcopy(X)
        _X = []
        len = X.shape[1]
        for i in range(len):
            X, temp = np.split(X, (-1,), axis=1)
            temp = temp.reshape(-1)
            _X.extend([temp])
        _X.reverse()
        self._X, self.Y = _X, Y.reshape(-1)
        self.b = (self.Y - self.expantionPoint[-1])[:-1]
        y_pred = self._calY(f_taylor, self._x, self._X)
        self.nihe_flag = False
        if mean_squared_error(self.Y, y_pred) < 0.01:
            self.nihe_flag = True
        self._mid_left, self._mid_right = 0, 0
        self._x_left, self._x_right = 0, 0

    def judge_Low_polynomial(self):
        f_taylor = str(self.f_taylor).split()
        try:
            self.bias = float(f_taylor[-2] + f_taylor[-1])
        except BaseException:
            self.bias = 0.
        f_low_taylor = ''
        if self.cal_power_expr(f_taylor[0]) <= 4:
            f_low_taylor += f_taylor[0]
        for i in range(2, len(f_taylor), 2):
            if self.cal_power_expr(f_taylor[i - 1] + f_taylor[i]) <= 4:
                f_low_taylor += f_taylor[i - 1] + f_taylor[i]
        self.f_low_taylor = sympify(f_low_taylor)
        print(f_low_taylor)
        y_pred_low = self._calY(self.f_low_taylor, self._x, self._X)
        self.low_nmse = mean_squared_error(self.Y, y_pred_low)
        if self.low_nmse < 1e-5:
            return True
        else:
            return False

    def judge_Bound(self):
        y_bound, var_bound = [], []
        _X = copy.deepcopy(self._X)
        for i in range(len(_X)):
            _X[i].sort()
            var_bound.extend([_X[i][0], _X[i][-1]])
        _Y = self.Y.reshape(-1)
        _Y.sort()
        y_bound.extend([_Y[0], _Y[-1]])
        return [y_bound, var_bound]

    def change_XToX(self, _X):
        len1 = len(_X)
        len2 = len(_X[0])
        X = np.array(_X[0])
        X = X.reshape(len(_X[0]), 1)
        for i in range(1, len1):
            temp = np.array(_X[i]).reshape(len2, 1)
            X = np.concatenate((X, temp), axis=1)
        return X

    def _cal_add_separability(self, multi_flag=False, expantionpoint=None):
        f_taylor = str(copy.deepcopy(self.f_taylor))
        for i in range(len(self._x_right)):
            f_taylor = f_taylor.replace(str(self._x_right[i]), str(self._mid_right[i])) + '-(' + str(self.bias) + ')'
        self.f_left_taylor = f_taylor
        f_taylor = str(copy.deepcopy(self.f_taylor))
        for i in range(len(self._x_left)):
            f_taylor = f_taylor.replace(str(self._x_left[i]), str(self._mid_left[i]))
        self.f_right_taylor = f_taylor

        self.f_left_taylor = sympify(self.f_left_taylor)
        self.f_right_taylor = sympify(self.f_right_taylor)
        Y_left = self._calY(self.f_left_taylor, self._x_left, self._X_left)
        Y_right = self._calY(self.f_right_taylor, self._x_right, self._X_right)
        self.X_left = self.change_XToX(self._X_left)
        self.X_right = self.change_XToX(self._X_right)
        if multi_flag:
            f_taylor = str(copy.deepcopy(self.f_taylor))
            for i in range(len(self._x)):
                f_taylor = f_taylor.replace(str(self._x[i]), str(expantionpoint[i]))
            self.f_mid = eval(f_taylor)
        len_Y = len(Y_left)
        Y_left, Y_right = np.array(Y_left), np.array(Y_right)
        Y = self.Y.reshape(len_Y, 1)
        self.Y_left = Y_left.reshape(len_Y, 1)
        try:
            if multi_flag:
                self.Y_right = Y_right.reshape(len_Y, 1)
                self.Y_right_temp = Y / self.Y_left
            else:
                self.Y_right = Y - self.Y_left
        except BaseException:
            self.Y_right_temp = Y_right.reshape(len_Y, 1)
        return None

    def judge_additi_separability(self, f_taylor=None, taylor_log_flag=False):
        '''additive separability discrimination'''
        for i in range(len(self._x) - 1):
            _x = copy.deepcopy(self._x)
            _x_left = _x.pop(i)
            _x_right = _x
            if f_taylor == None:
                f_taylor = self.f_taylor
            f_taylor_split = str(f_taylor).split()
            f_temp = []
            for subF in f_taylor_split:
                if any([str(_x_left) in subF and str(_x_right[j]) in subF for j in range(len(_x_right))]):
                    f_temp = f_temp[:-1]
                else:
                    f_temp.append(subF)
            f = ''
            for temp in f_temp:
                f += temp
            f = sympify(f)
            y_pred = self._calY(f, self._x, self._X)
            if taylor_log_flag:
                nmse = mean_squared_error(self.Y_log, y_pred)
            else:
                nmse = mean_squared_error(self.Y, y_pred)
            if nmse > 0.001:
                continue
            else:
                _x = copy.deepcopy(self._x)
                self._x_left = [_x[i]]
                _x.pop(i)
                self._x_right = _x
                expantionpoint = copy.deepcopy(self.expantionPoint).tolist()
                self._mid_left = [expantionpoint[i]]
                expantionpoint.pop(i)
                self._mid_right = expantionpoint[:-1]
                _X = copy.deepcopy(self._X)
                self._X_left = [_X.pop(i)]
                try:
                    a = _X[0].shape[0]
                except BaseException:
                    _X = [_X]
                self._X_right = _X
                self._cal_add_separability()
                print('addition separable')
                return True

    def judge_multi_separability(self):
        '''multiplicative separability discrimination'''
        _expantionpoint = copy.deepcopy(self.expantionPoint)
        if self.expantionPoint[-1] == 0:
            _expantionpoint = _expantionpoint + 0.2
        for i in range(len(self._x) - 1):
            f_taylor = self.f_taylor
            _x = copy.deepcopy(self._x)
            self._x_left = [_x[i]]
            _x.pop(i)
            self._x_right = _x
            expantionpoint = copy.deepcopy(_expantionpoint).tolist()
            self._mid_left = [expantionpoint[i]]
            expantionpoint.pop(i)
            self._mid_right = expantionpoint[:-1]
            _X = copy.deepcopy(self._X)
            self._X_left = [_X.pop(i)]
            try:
                a = _X[0].shape[0]
            except BaseException:
                _X = [_X]
            self._X_right = _X
            self._cal_add_separability(multi_flag=True, expantionpoint=expantionpoint)
            nmse = mean_squared_error(self.Y, self.Y_left * self.Y_right / self.f_mid)
            if nmse < 0.001:
                print('multiplication separable')
                print('multi_nmse = ', nmse)
                self.Y_right = self.Y_right_temp
                return True
            else:
                try:
                    return self.judge_additi_separability(f_taylor=self.f_taylor_log, taylor_log_flag=True)
                except BaseException:
                    return False

        return False

    def judge_parity(self):
        '''
        return：non odd non even function：-1
                odd function：1
                even function：2
        '''
        if self.nihe_flag:
            if self.bias != 0:
                f_taylor = str(self.f_taylor).split()[:-2]
            else:
                f_taylor = str(self.f_taylor).split()
            Y = self.Y - self.bias
            f_odd, f_even = '', ''
            if self.cal_power_expr(f_taylor[0]) % 2 == 1:
                f_odd += f_taylor[0]
            else:
                f_even += f_taylor[0]
            for i in range(2, len(f_taylor), 2):
                if self.cal_power_expr(f_taylor[i - 1] + f_taylor[i]) % 2 == 1:
                    f_odd += f_taylor[i - 1] + f_taylor[i]
                else:
                    f_even += f_taylor[i - 1] + f_taylor[i]
            f_odd, f_even = sympify(f_odd), sympify(f_even)
            Jishu, Oushu, nmse_odd, nmse_even = False, False, 0, 0
            y_pred = self._calY(f_odd, self._x, self._X)
            nmse_odd = mean_squared_error(Y, y_pred)
            y_pred = self._calY(f_even, self._x, self._X)
            nmse_even = mean_squared_error(Y, y_pred)
            print('NMSE of parity function', nmse_odd, nmse_even, sep='\n')
            if nmse_odd < 0.001:
                Jishu = True
            if nmse_even < 0.001:
                Oushu = True
            if Jishu == True and Oushu == False:
                print('Odd function！！！')
                self.parity_flag = True
                return 1
            elif Jishu == False and Oushu == True:
                self.parity_flag = True
                print('even function！！！')
                return 2
        print('non odd non even function！！！')
        return -1


def cal_Taylor_features(varNum, dataSet, Y=None):
    '''qualified_list = [low_high_target_bound, low_high_var_bound,bias,partity,monity]'''
    qualified_list = []
    low_polynomial = False
    loopNum = 0
    Metric = []
    while True:
        metric = Metrics(varNum=varNum, dataSet=dataSet)
        loopNum += 1
        Metric.append(metric)
        if loopNum == 1:
            break
    Metric.sort(key=lambda x: x.nmse)
    metric = Metric[0]
    print('NMSE of polynomial and lower order polynomial after sorting', metric.nmse, metric.low_nmse)
    if metric.nmse < 0.1:
        metric.nihe_flag = True
    else:
        print('Fitting failed')
    if metric.judge_Low_polynomial():
        print('The result is a low order polynomial')
        low_polynomial = True

    '''
    add_seperatity = metric.judge_additi_separability()
    multi_seperatity = metric.judge_multi_separability()

    qualified_list.extend(metric.judge_Bound()) 
    # qualified_list.extend([1,1,1,1])
    qualified_list.append(metric.f_low_taylor)
    qualified_list.append(metric.low_nmse) 
    qualified_list.append(metric.bias)  
    qualified_list.append(metric.judge_parity())
    qualified_list.append(metric.judge_monotonicity())
    # qualified_list.append(metric.di_jian_flag)
    print('qualified_list = ',qualified_list)
    # X,Y = metric.X, metric.change_Y(Y)
    return metric.nihe_flag,low_polynomial,qualified_list,metric.change_Y(Y)     
    '''


if __name__ == '__main__':
    Global()
    fileName = "D:\PYcharm_program\Test_everything\Bench_0.15\BenchMark_44.tsv"
    fileName = "example.tsv"
    X_Y = np.loadtxt(fileName, dtype=np.float, skiprows=1)
    for i in [1]:
        # cal_Taylor_features(varNum=2, fileName="example.tsv")
        cal_Taylor_features(varNum=2,dataSet=X_Y )

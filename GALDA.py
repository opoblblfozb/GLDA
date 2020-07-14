import warnings
import math
from scipy.special import psi
import numpy as np
import pickle
from math import gamma
from scipy.special import loggamma
from pathlib import Path
import networkx as nx
from networkx.generators import classic
from networkx.generators import small
from networkx.generators import ego
from scipy import stats
LOWERLIMIT = 1.0e-300

warnings.simplefilter('error', category=RuntimeWarning)

class GALDA():
    def __init__(self, data, classnum):
        ### classnum
        self.classnum = classnum

        ### data
        self.data = data # list
        self.M = len(data)
        self.N_m = []
        for i in range(self.M):
            datam = self.data[i]
            self.N_m.append(len(datam))
        self.D = len(data[0][0])

        ### prior_params
        # Ditichlet prior param alpha
        self.alpha = np.ones(self.classnum) 
        # Gauss Wishart prior param m, beta, r, W
        scale = 1
        self.m = scale*np.random.rand(self.D)
        self.beta = 1
        self.r = self.D + 1
        self.W = np.eye(self.D)

        ### variational(V)_params
        # z's param theta_m_n size:m*n*k
        self.Vtheta_m_n = []
        for i in range(self.M):
            Vthetas_m = []
            for j  in range(self.N_m[i]):
                probs = np.random.rand(self.classnum)
                probs = probs / np.sum(probs)
                Vthetas_m.append(probs.tolist())
            self.Vtheta_m_n.append(Vthetas_m)
        # theta_m's param alpha_m size:m*classnum
        self.Valpha_m = np.ones((self.M, self.classnum))
        # mu's param m(size: classnum*D), beta(size:k)
        self.Vm = np.random.rand(self.classnum, self.D)
        self.Vbeta = np.ones(self.classnum)
        # Lsigma's param W, v
        self.VW = []
        for _ in range(self.classnum):
            self.VW.append(np.eye(self.D))
        self.VW = np.array(self.VW)
        self.Vr = np.array([self.D + 2 for _ in range(self.classnum)])

    def CalucuSecond(self, forward, middle, back):
        res = np.dot(middle, back)
        res = np.dot(forward, res)
        return res

    def train(self, Maxiter=1000, th=1.0e-4):
        for _ in range(Maxiter):
            beforELBO = float('inf')

            print('iteration....', _)
            self.Renew_qz()
            self.Renew_qmusigma()
            self.Renew_qtheta()
            # print('----vm------')
            # print(self.Vm)    
            # print('----VW------')
            # for k in range(self.classnum):
            #     print(self.Vr[k] * self.VW[k])
            ELBO = self.CaluELBO()
            print('ELBO: {}'.format(ELBO))
    
            if abs(beforELBO - ELBO) < th:
                print('converged')
                break

            beforELBO = ELBO
    
    ######################## about qz
    def Renew_qz(self):
        for m in range(self.M):
            for n in range(self.N_m[m]):
                theta = self.caluculate_exptheta(m=m,n=n)
                theta = np.array(theta) / np.sum(theta)
                theta = np.where(theta<LOWERLIMIT, LOWERLIMIT, theta)
                self.Vtheta_m_n[m][n] = theta.tolist()
        
    def caluculate_exptheta(self, m, n):
        datamn = self.data[m][n]

        part1 = psi(self.Valpha_m[m]) - psi(np.sum(self.Valpha_m[m]))
        part2 = []
        part3 = []
        part4 = []
        part5 = []
        for k in range(self.classnum):
            tmp2 = (-0.5)*self.Vr[k]*self.CalucuSecond(datamn, self.VW[k], datamn)
            part2.append(tmp2)
            
            tmp3 = (-0.5)*self.Vr[k]*self.CalucuSecond(self.Vm[k], self.VW[k], self.Vm[k])
            tmp3 += (-0.5)*(self.D/self.Vbeta[k])
            part3.append(tmp3)

            tmp4 = self.Vr[k]*self.CalucuSecond(np.array(datamn), self.VW[k], self.Vm[k])
            part4.append(tmp4)

            tmp5 = sum([psi((self.Vr[k]+1-d)*0.5) for d in range(1, self.D+1)])
            tmp5 += self.D*np.log(2)
            tmp5 += np.log(np.linalg.det(self.VW[k]))
            part5.append(0.5*tmp5)

        res = np.array(part1) + np.array(part2) + \
                np.array(part3) + np.array(part4) + np.array(part5)
        res = np.exp(res)
        return res

    # def caluculate_theta2(self, i, j):
    #     res = 0
    #     dataij = self.data[i][j]
    #     ### Expect logthetam_k part1
    #     res += psi(self.Valpha_m[i]) - psi(np.sum(self.Valpha_m[i]))
    #     ### Expect Vmuk*Lsigmak*Vmuk part3
    #     term = []
    #     for k in range(self.classnum):
    #         val = self.Vr[k] * self.CalucuSecond(self.Vm[k],
    #                                              self.VW[k], self.Vm[k])
    #         val += self.D/self.Vbeta[k]
    #         term.append(0.5*val)
    #     res -= np.array(term)
    #     # ### Expect Lgimak*Vmuk part4
    #     term = []
    #     for k in range(self.classnum):
    #         val = self.Vr[k] * np.dot(self.VW[k], self.Vm[k])
    #         val = np.dot(np.array(dataij), val)
    #         term.append(val)
    #     res += np.array(term)
    #     ### Expect Sigma part2
    #     term = []
    #     for k in range(self.classnum):
    #         expectsigma = self.Vr[k]*self.VW[k]
    #         val = self.CalucuSecond(dataij, expectsigma, dataij)
    #         term.append(0.5*val)
    #     res -= np.array(term)
    #     ### Expect logsigma part5
    #     term = []
    #     for k in range(self.classnum):
    #         val = np.sum([psi((self.Vr[k]+1-d)*0.5) for d in range(1, self.D+1)])#psiの中身が０以下になるとエラー
    #         val += self.D*np.log(2)
    #         val += np.log(np.linalg.det(self.VW[k]))
    #         term.append(0.5*val)
    #     res += np.array(term)
    #     res = np.exp(res)
    #     return res
                                                            

    ######################## about qmusigma
    def Renew_qmusigma(self):
        self.Renew_qmu()
        self.Renew_qsigma()

    def Renew_qmu(self):
        ### renew_Vbeta
        sumbymn = np.array([0 for _ in range(self.classnum)], dtype='float')
        for i in range(self.M):
            thetamnk = np.array(self.Vtheta_m_n[i])
            sumbyn = np.sum(thetamnk, axis=0)
            sumbymn += sumbyn
        self.Vbeta = sumbymn + self.beta
        ### renew_Vm
        betam = self.m*self.beta
        for k in range(self.classnum):
            weitedsum = np.array([0 for _ in range(self.D)], dtype='float')

            for i in range(self.M):
                for j in range(self.N_m[i]):
                    val = self.Vtheta_m_n[i][j][k]*np.array(self.data[i][j])
                    weitedsum += val

            self.Vm[k] = (weitedsum + betam) / self.Vbeta[k]
            
    def Renew_qsigma(self):
        ### renew_VW
        for k in range(self.classnum):
            VinvW = np.zeros((self.D, self.D))
            weitedsum = np.zeros((self.D, self.D))
            for i in range(self.M):
                for j in range(self.N_m[i]):
                    outer = np.outer(self.data[i][j], self.data[i][j])
                    weitedsum += self.Vtheta_m_n[i][j][k]*outer
            VinvW += weitedsum
            VinvW += self.beta*np.outer(self.m, self.m)
            VinvW -= self.Vbeta[k]*np.outer(self.Vm[k], self.Vm[k])
            VinvW += np.linalg.inv(self.W)

            self.VW[k] = np.linalg.inv(VinvW)
        ### renew_Vr
        sumbymn = np.array([0 for _ in range(self.classnum)], dtype='float')
        for i in range(self.M):
            thetamnk = np.array(self.Vtheta_m_n[i])
            sumbyn = np.sum(thetamnk, axis=0)
            sumbymn += sumbyn
        self.Vr = sumbymn + self.r

    ######################## about qtheta
    def Renew_qtheta(self):
        for i in range(self.M):
            thetamnk = np.array(self.Vtheta_m_n[i])
            sumbyn = np.sum(thetamnk, axis=0)
            self.Valpha_m[i] = sumbyn + self.alpha

    ######################## about ELBO
    def CaluELBO(self):
        minusKLtheta = self.minusKLtheta()
        minusKLphi = self.minusKLphi()
        exps = self.Exps()
        res = minusKLtheta + minusKLphi + exps
        print('ELBO is divided fllowing')
        print('minusKLtheta', minusKLtheta)
        print('minusKLphi', minusKLphi)
        print('exps', exps)
        return res
    
    ########### about minusKLtheta
    def minusKLtheta(self):
        res = 0
        for m in range(self.M):
            res += self._logDirCd(self.alpha)
            res -= self._logDirCd(self.Valpha_m[m])

            for k in range(self.classnum):
                alphapart = self.alpha[k] - self.Valpha_m[m, k]
                psipart = psi(self.Valpha_m[m,k]) - psi(np.sum(self.Valpha_m[m]))
                res += alphapart*psipart
        return res

    def _logDirCd(self, al):
        res = loggamma(sum(al))
        res -= sum([loggamma(a) for a in al])
        return res

    # def minusKLtheta(self):
    #     res = 0
    #     for m in range(self.M):
    #         res += self._logDirCd(self.alpha)
    #         res -= self._logDirCd(self.Valpha_m[m])

    #         for n in range(self.N_m[m]):
    #             for k in range(self.classnum):
    #                 theta = self.Vtheta_m_n[m][n][k]
    #                 Explogtheta = psi(self.Valpha_m[m, k])
    #                 Explogtheta -= psi(np.sum(self.Valpha_m[m]))
    #                 res += theta*Explogtheta
    #     return res

    ########### about minusKLphi() kokoが諸悪の根源 
    def minusKLphi(self):
        res = 0
        for k in range(self.classnum):
            res += self.normpart(k)
            res += self.wishpart(k)
        return res

    def normpart(self, k):
        part1 = self.Vr[k]*self.CalucuSecond(self.Vm[k], self.VW[k], self.Vm[k])
        part1 += self.D/self.Vbeta[k]
        part1 *= 0.5*(self.Vbeta[k] - self.beta)
        part2 = -0.5*(self.beta*self.Vr[k]* \
                      self.CalucuSecond(self.m, self.VW[k], self.m))
        part3 = 0.5*self.Vbeta[k]*self.Vr[k]* \
                      self.CalucuSecond(self.Vm[k], self.VW[k], self.Vm[k])
        part4_1 = self.Vr[k]*np.dot(self.VW[k], self.Vm[k])
        part4_2 = (self.beta*self.m - self.Vbeta[k]*self.Vm[k])
        part4 = np.dot(part4_2, part4_1)
        part5 = (self.D/2)*(np.log(self.beta)-np.log(self.Vbeta[k]))
        res = part1 + part2 + part3 + part4 + part5
        return res

    def wishpart(self, k):
        part1 = sum([psi((self.Vr[k]+1-d)/2) for d in range(1, self.D+1)])
        part1 += self.D*np.log(2) + np.log(np.linalg.det(self.VW[k]))
        part1 *= (self.r-self.Vr[k])/2
        part2 = np.linalg.inv(self.VW[k]) - np.linalg.inv(self.W)
        part2 = np.dot(part2, self.Vr[k]*self.VW[k])
        part2 = 0.5 * np.trace(part2)
        part3 = -(self.r/2)*np.log(np.linalg.det(self.W))
        part4 = (self.Vr[k]/2)*np.log(np.linalg.det(self.VW[k]))
        part5 = (self.Vr[k] - self.r)*(self.D/2)*np.log(2)
        part6 = sum([loggamma((self.Vr[k]+1-d)/2) for d in range(1, self.D+1)])
        part6 -= sum([loggamma((self.r+1-d)/2) for d in range(1, self.D+1)])
        res = part1 + part2 + part3 + part4 + part5 + part6
        return res

    # def normpart(self, k):
    #     res = 0
    #     res -= 0.5*(self.beta-self.Vbeta[k])* \
    #             (self.CalucuSecond(self.Vm[k], self.Vr[k]*self.VW[k], self.Vm[k]) + \
    #              self.D/self.Vbeta[k])
    #     res += (self.D/2)*(np.log(self.beta) - np.log(self.Vbeta[k]))
    #     tmp = (self.beta*self.m + self.Vbeta[k]*self.Vm[k])
    #     res += self.CalucuSecond(tmp, self.Vr[k]*self.VW[k], self.m)
    #     res -= 2*self.CalucuSecond(self.Vbeta[k]*self.m,self.Vr[k]*self.VW[k],self.Vm[k])
    #     return res

    # def wishpart(self, k):
    #     exploglamdapart = sum([psi((self.Vr[k]+1-d)/2) for d in range(1, self.D+1)])
    #     exploglamdapart += self.D*np.log(2) + np.log(np.linalg.det(self.VW[k]))
    #     exploglamdapart *= (self.r-self.Vr[k])/2
    #     tracepart = np.linalg.inv(self.W) + np.linalg.inv(self.VW[k])
    #     tracepart = np.dot(tracepart, self.Vr[k]*self.VW[k])
    #     tracepart = -0.5*np.trace(tracepart)
    #     Cwpart = self._wishlogCw(self.r, self.W, self.D) \
    #                     - self._wishlogCw(self.Vr[k], self.VW[k], self.D)
    #     res = exploglamdapart + tracepart + Cwpart
    #     return res


    # def _wishlogCw(self, r, W, D):
    #     res = 0
    #     res -= (r/2)*np.log(np.linalg.det(W))
    #     res -= ((r*D)/2)*np.log(2)
    #     res -= ((D*(D-1))/4)*np.log(np.pi)
    #     # loggamma の中身が負になるとnan
    #     res -= sum([loggamma((r+1-d)/2) for d in range(1, self.D+1)])
    #     return res

    ########### about Exps() ... logqz, logpz, logpx koko
    def Exps(self):
        res = 0
        for m in range(self.M):
            for n in range(self.N_m[m]):
                for k in range(self.classnum):
                    theta = self.Vtheta_m_n[m][n][k]
                    logtheta = -np.log(theta)
                    explogtheta = psi(self.Valpha_m[m,k]) - \
                                        psi(np.sum(self.Valpha_m[m]))
                    expnorm = self._calcuexpnorm(m, n, k)
                    res += theta*(logtheta + explogtheta + expnorm)
        return res
    
    def _calcuexpnorm(self, m, n, k):
        r = 0
        data = self.data[m][n]
        r -= 0.5*self.CalucuSecond(data, self.Vr[k]*self.VW[k], data)
        r -= 0.5*self.CalucuSecond(self.Vm[k], self.Vr[k]*self.VW[k], self.Vm[k])
        r -= 0.5*(self.D/self.Vbeta[k])
        r += self.CalucuSecond(data, self.Vr[k]*self.VW[k], self.Vm[k])
        r += sum([psi((self.Vr[k] + 1 - d)/2) for d in range(1, self.D+1)])
        r += np.log(np.linalg.det(self.VW[k]))
        r += self.D*np.log(2)
        r -= (0.5)*self.D*np.log(2*np.pi)
        return r

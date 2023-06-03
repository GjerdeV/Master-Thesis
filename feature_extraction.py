import numpy as np
from ssqueezepy import cwt, ssq_cwt
from ssqueezepy.visuals import imshow
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import kurtosis, skew
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
from scipy.linalg import lu, svd


# time        = np.arange(0, 10, 0.1)
# amplitude   = np.sin(time)

def featext(data):
    t = np.arange(0, 5, 0.002, dtype='float64')
    feature_list = []
    for m in range(len(data)):
        sig = data[m]
        # sig = epochs[1].iloc[:,m].to_numpy()
        Wx, _ = cwt(sig, 'morlet', t=t)
        # feature_list.append(Wx)
        rms = np.sqrt(np.mean(sig**2, axis=-1))
        var = np.var(sig, axis=-1)
        minim = np.min(sig, axis=-1)
        maxim = np.max(sig, axis=-1)
        argminim = np.argmin(sig, axis=-1)
        argmaxim = np.argmax(sig, axis=-1)
        skews = skew(sig, axis=-1)
        kurtos = kurtosis(sig, axis=-1)
        # aWx = np.abs(Wx)
        # nx, ny = Wx.shape
        # wave = Wx.reshape((nx*ny))
        # feature = np.concatenate((wave, np.array([rms, var, minim, maxim, argminim, argmaxim, skews, kurtos])), axis=-1)
        # feature = sel.fit_transform(np.abs(Wx))
        # p, l, u = lu(Wx)
        # e, v = np.linalg.eig(Wx)
        # e.real
        # v.real
        U,_,_ = svd(Wx)
        U.real
        # feature = []
        feature_list.append(U)
    return feature_list

def featest(data):
    t = np.arange(0, 5, 0.002, dtype='float64')
    feature_list = []
    for m in range(len(data)):
        sig = data[m]
        Tx, *_ = ssq_cwt(sig, 'morlet', t=t)
        pca = PCA(n_components=4)
        pca.fit(np.abs(Tx))
        feature_list.append(pca.singular_values_)
        
    return feature_list

# Tx, *_ = ssq_cwt(amplitude, 'morlet', t=time, scales='log-piecewise')

# # Tx, *_ = ssq_cwt(s, wavelet, t=t)
# #%%# 'cheat' a little; could use boundary wavelets instead (not implemented)
# aTxz = np.abs(Tx)[:, len(time) // 8:]
# imshow(aTxz, abs=1, title="abs(SSWT(s(t)))", show=1, cmap='bone')
# imshow(Wx)

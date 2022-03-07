import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from cpython cimport PyObject, Py_INCREF, Py_DECREF


def mcepalpha(fs, start=0.0, stop=1.0, step=0.001, num_points=1000):
    """Compute appropriate frequency warping parameter given a sampling frequency

    It would be useful to determine alpha parameter in mel-cepstrum analysis.

    The code is traslated from https://bitbucket.org/happyalu/mcep_alpha_calc.

    Parameters
    ----------
    fs : int
        Sampling frequency

    start : float
        start value that will be passed to numpy.arange. Default is 0.0.

    stop : float
        stop value that will be passed to numpy.arange. Default is 1.0.

    step : float
        step value that will be passed to numpy.arange. Default is 0.001.

    num_points : int
        Number of points used in approximating mel-scale vectors in fixed-
        length.

    Returns
    -------
    alpha : float
        frequency warping paramter (offen denoted by alpha)

    See Also
    --------
    pysptk.sptk.mcep
    pysptk.sptk.mgcep

    """
    
    def _melscale_vector(fs, length):
        step = (fs / 2.0) / length
        melscalev = 1000.0 / np.log(2) * np.log(1 + step * np.arange(0, length) / 1000.0)
        return melscalev / melscalev[-1]
    
    def _warping_vector(alpha, length):
        step = np.pi / length
        omega = step * np.arange(0, length)
        num = (1 - alpha * alpha) * np.sin(omega)
        den = (1 + alpha * alpha) * np.cos(omega) - 2 * alpha
        warpfreq = np.arctan(num / den)
        warpfreq[warpfreq < 0] += np.pi
        return warpfreq / warpfreq[-1]
    
    def _rms_distance(v1, v2):
        d = v1 - v2
        return np.sum(np.abs(d * d)) / len(v1)

    alpha_candidates = np.arange(start, stop, step)
    mel = _melscale_vector(fs, num_points)
    distances = [
        _rms_distance(mel, _warping_vector(alpha, num_points))
        for alpha in alpha_candidates
    ]
    return alpha_candidates[np.argmin(distances)]


cdef void freqt(const double *c, double *wc, double *prev, int c_size, double alpha, int order) nogil:
    cdef int i, j
    for i in range(-(c_size - 1), 1):  # -(c_size - 1) 〜 0
        memcpy(prev, wc, sizeof(double)*(order + 1))
        if order >= 0:
            wc[0] = c[-i] + alpha * prev[0]
        if order >= 1:
            wc[1] = (1.0 - alpha ** 2) * prev[0] + alpha * prev[1]
        for j in range(2, order + 1):  # 2 ~ order -> length(wc) -> order + 1
            wc[j] = prev[j - 1] + alpha * (prev[j] - wc[j - 1])


def spec2mcep(np.ndarray[np.float64_t, ndim=2, mode="c"] spmat, int order, double alpha):
    """
    calucurate mel-cepstrum from spectrogram matrix
    """
    cdef int fftsize = (spmat.shape[1] - 1) * 2
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] logspmat = np.log(spmat)
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] c_mat = np.real(np.fft.irfft(logspmat, fftsize))

    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] wc_mat = np.zeros(
        (c_mat.shape[0], order + 1),dtype=np.double)
    
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] c_vec, wc_vec, prev
    prev = np.zeros(wc_mat.shape[1], dtype=np.float64)
    for i in range(c_mat.shape[0]):
        # ToDo: change ndarray to typed memoryview
        wc_vec = wc_mat[i]
        c_vec = c_mat[i]
        c_vec[0] /= 2.0
        freqt(<double *>c_vec.data, <double *>wc_vec.data, <double *>prev.data, c_vec.size, alpha, order)
    return wc_mat

def mcep2spec(np.ndarray[np.float64_t, ndim=2, mode="c"] mcepmat, double alpha, int fftsize):
    cdef int fsize = fftsize >> 1
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] c = np.zeros((mcepmat.shape[0], fsize + 1), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] prev = np.zeros(fsize + 1, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] ret = np.zeros((mcepmat.shape[0], fftsize), dtype=np.float64)
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] c_vec, mcep_vec, ret_vec
    for i in range(mcepmat.shape[0]):
        # ToDo: change ndarray to typed memoryview
        c_vec = c[i]
        mcep_vec = mcepmat[i]
        ret_vec = ret[i]
        freqt(<double *>mcep_vec.data, <double *>c_vec.data, <double *>prev.data, mcep_vec.size, -alpha, fsize)
        c_vec[0] *= 2.0
        np.copyto(ret_vec[:fsize + 1], c_vec)
        for i in range(fsize):
            ret_vec[fftsize - i - 1] = ret_vec[i+1]
    return np.exp(np.real(np.fft.rfft(ret)))



cdef double[:] pade_coef4 = np.array([1.0, 4.999273e-1, 1.067005e-1, 1.170221e-2, 5.656279e-4])
cdef double[:] pade_coef5 = np.array([1.0, 4.999391e-1, 1.107098e-1, 1.369984e-2, 9.564853e-4, 3.041721e-5])

cdef class Filter:
    cdef double alpha
    cdef int order
    cdef double[:] delay

    def __cinit__(self, int order, double alpha):
        self.order = order
        self.alpha = alpha
        self.delay = np.zeros(order + 1, dtype=np.float64)

    cdef filter(self, double x, np.ndarray[np.float64_t, ndim=1, mode="c"] coefficients):
        cdef double result = 0.0
        self.delay[0] = x
        self.delay[1] = (1.0 - self.alpha ** 2) * self.delay[0] + self.alpha * self.delay[1]
        cdef int i
        for i in range(2, coefficients.size):
            self.delay[i] = self.delay[i] + self.alpha * (self.delay[i + 1] - self.delay[i - 1])
            result += self.delay[i] * coefficients[i]
        if coefficients.shape[0] == 2:
            result += self.delay[1] * coefficients[1]
        for i in range(-(self.delay.size - 1), -1):
            i = -i
            self.delay[i] = self.delay[i - 1]
        return result

cdef class CascadeFilter:
    cdef int pade_order, order, filter_num
    cdef double alpha
    cdef double[:] delay
    cdef double[:] pade_coefficients
    cdef PyObject **filters

    def __cinit__(self, int order, double alpha, int pade_order):
        self.pade_order = pade_order
        self.filter_num = pade_order + 1
        self.order = order
        self.alpha = alpha
        self.delay = np.zeros(self.filter_num, dtype=np.float64)
        self.filters = <PyObject **>malloc(sizeof(PyObject *) * self.filter_num)
        for i in range(self.filter_num):
            filt = Filter(order, alpha)
            Py_INCREF(filt)
            self.filters[i] = <PyObject *>filt
        if pade_order == 4:
            self.pade_coefficients = pade_coef4
        else:
            self.pade_coefficients = pade_coef5

    def __dealloc__(self):
        cdef int i
        for i in range(self.filter_num):
            Py_DECREF(<object>self.filters[i])
        free(self.filters)

    cdef filter(self, double x, np.ndarray[np.float64_t, ndim=1, mode="c"] coefficients):
        cdef double result = 0.0
        cdef double feedback = 0.0
        cdef int i
        for i in range(-(len(self.pade_coefficients) - 1), 0):
            i = -i
            self.delay[i] = (<Filter>self.filters[i]).filter(self.delay[i - 1], coefficients)
            val = self.delay[i] * self.pade_coefficients[i]
            if i % 2 == 1:
                feedback += val
            else:
                feedback -= val
            result += val
        self.delay[0] = feedback + x
        result += self.delay[0]
        return result

cdef class MLSAFilter(object):
    cdef int pade_order, order
    cdef double alpha
    cdef CascadeFilter f1
    cdef CascadeFilter f2

    def __init__(self, int order, double alpha, int pade_order):
        assert pade_order == 4 or pade_order == 5, "order of pade must be 4 or 5."
        self.f1 = CascadeFilter(2, alpha, pade_order)
        self.f2 = CascadeFilter(order + 1, alpha, pade_order)

    def filter(self, x, coefficients):
        coef = np.array([0, coefficients[1]])
        return self.f2.filter(self.f1.filter(x, coef), coefficients)

class Synthesizer(object):        
    def __init__(self, order, alpha, pade_order):
        self.alpha = alpha
        self.order = order
        self.prev_coef = np.zeros((order, ), dtype=np.dtype('float64'))
        self.cur_coef = np.zeros((order, ), dtype=np.dtype('float64'))

        self.filter = MLSAFilter(order, alpha, pade_order)


    def _mcep2coef(self, mcep):
        self.cur_coef[:] = mcep[:]
        for i in range(0, mcep.size - 1)[::-1]:
            self.cur_coef[i] = self.cur_coef[i] - self.alpha * self.cur_coef[i+1]
        
        
    def __call__(self, pulse, mcep):
        self._mcep2coef(mcep)
        
        ret = np.zeros_like(pulse)
        slope = (self.cur_coef - self.prev_coef) / len(pulse)
        for i in range(len(pulse)):
            self.prev_coef[:] = self.prev_coef + slope
            scaled = pulse[i] * np.exp(self.prev_coef[0])
            ret[i] = self.filter.filter(scaled, self.prev_coef)
        return ret

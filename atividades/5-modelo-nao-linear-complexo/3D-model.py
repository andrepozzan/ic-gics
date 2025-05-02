from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from numba import njit, prange
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.cm as cm

loaded_data = loadmat('in_out_SBRT2_direto.mat')
in_training   = loaded_data['in_extraction']
out_training  = loaded_data['out_extraction']
in_validation = loaded_data['in_validation']
out_validation= loaded_data['out_validation']

@njit(parallel=True)
def estimatedValueWithComplex(x_data, coef_matrix, P, M):
    N = len(x_data)
    y_est = np.empty(N, dtype=np.complex128)
    for n in prange(N):
        estimated_value = 0.0 + 0.0j
        for p in range(1, P+1):
            for m in range(M+1):
                idx   = max(0, n-m)
                power = 2*p - 2
                term  = (np.abs(x_data[idx])**power) * x_data[idx]
                estimated_value  += term * coef_matrix[p-1, m]
        y_est[n] = estimated_value
    return y_est


def unpackComplexCoefficients(x_real, P, M):
    half = len(x_real) // 2
    real_parts = x_real[:half]
    imag_parts = x_real[half:]
    complex_coef = real_parts + 1j*imag_parts
    return complex_coef.reshape((P, M+1))


def mpResiduals(x_real, x_data, y_data, P, M):
    coef_matrix = unpackComplexCoefficients(x_real, P, M)
    y_est = estimatedValueWithComplex(x_data, coef_matrix, P, M)
    resid = y_data.ravel() - y_est
    return np.concatenate((resid.real, resid.imag))


def generateInitialComplex(P, M, mean, std):
    shape = P * (M + 1)
    real_part = np.random.normal(loc=mean.real, scale=std, size=shape)
    imag_part = np.random.normal(loc=mean.imag, scale=std, size=shape)
    return real_part + 1j * imag_part


def calcMethodByPandM(P, M):
    initial_complex = generateInitialComplex(P, M, mean=np.mean(in_training), std=np.std(in_training))
    initial_real = np.concatenate((initial_complex.real, initial_complex.imag))
    res = least_squares(mpResiduals, initial_real,
                        args=(in_training.ravel(), out_training.ravel(), P, M),
                        verbose=0)
    coef = unpackComplexCoefficients(res.x, P, M)
    y_val_est = estimatedValueWithComplex(in_validation.ravel(), coef, P, M)
    err_vec = np.abs(out_validation.ravel() - y_val_est)
    max_err = np.max(err_vec)
    return (P, M, max_err)


def runParallelEvaluation(P_MAX, M_MAX):
    params = [(p, m) for p in range(1, P_MAX+1) for m in range(1, M_MAX+1)]
    results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(calcMethodByPandM, p, m): (p, m) for p, m in params}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Progresso P×M"):
            results.append(future.result())
    return results

P_MAX, M_MAX = 10, 5
results = runParallelEvaluation(P_MAX, M_MAX)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmap = cm.viridis
for idx, (p, m, err) in enumerate(results):
    color = cmap(idx / len(results))
    ax.scatter(p, m, err, c=[color], marker='o', s=30)
ax.set_xlabel('P')
ax.set_ylabel('M')
ax.set_zlabel('Max Error')
ax.set_title('Validação: Max Error vs P e M')
ax.set_xticks(range(1, P_MAX+1))
ax.set_yticks(range(1, M_MAX+1))
plt.tight_layout()
plt.show()

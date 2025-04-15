import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Dummy training loop (replace with real processed data)


def model_f(x, a, b, c, d, e, f):
    return a + b*(x-30) + c*(x-30)**2 + d*(x-30)**3 + e*(x-30)**4 + f*(x-30)**5


if __name__ == '__main__':
    baseDat = np.array([-12.0, -11.8, -11.4, -10.5, -10.0, -8.5, -7.0, -6.0, -5.5, -5.2, -5.0, -4.8, -4.0, -3.0, -2,
               -0.4, 0.1, 2, 4, 5, 6, 8, 9.5, 10])
    targetDat = np.array([0.0, 0.009, 0.012, 0.018, 0.02, 0.1, 0.25, 0.43, 0.48, 0.5, 0.52, 0.51,
                          0.49, 0.485, 0.48, 0.5, 0.51, 0.53, 0.55, 0.56, 0.55, 0.52, 0.513, 0.51])

    popt, pcov = curve_fit(model_f,baseDat, targetDat, p0=[3, 2, -16, 0.3, 0.5, 0.01])
    ao, bo, co, do, eo, fo = popt
    x_model = np.linspace(min(baseDat), max(baseDat), 100)
    print(popt)
    y_model = model_f(x_model, ao, bo, co, do, eo, fo)
    plt.scatter(baseDat, targetDat)
    plt.plot(x_model, y_model, color='r')
    plt.show()
    # Shift baseDat values (to avoid issues with very small/negative inputs if log used)

    # Generate powerDat: each row is [power^0, power^1, ..., power^11]


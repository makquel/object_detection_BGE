import numpy as np
import matplotlib.pyplot as plt

def f_x(x):
    return np.power(x + 2, 2) - 16 * np.exp(-np.power((x - 2), 2))

#Vejamos a sua forma no intervalo [-8, 8]
def main_plot():
    x = np.arange(-8, 8, 0.001)
    y = map(lambda u: f_x(u), x)
    plt.plot(x, list(y))
    plt.title('Cost function f(x)')
	
plt.grid(True)
main_plot()
plt.show()

def grad_f_x(x):
    return (2 * x - 4) - 16 * (-2 * x + 4) * np.exp(-np.power(x - 2, 2))

def gradient_descent(x0, func, grad):
    #
    precision = 0.001
    #Learning rate
    learning_rate = 0.0001
    #
    max_iter = 10000
    x_new = x0
    res = []
    for i in range(max_iter):
        x_old = x_new
        #Vamos usar B = 1
        x_new = x_old - learning_rate * grad(x_old)
        f_x_new = func(x_new)
        f_x_old = func(x_old)
        res.append([x_new, f_x_new])
        #print(f_x_new - f_x_old)
        if(abs (f_x_new - f_x_old) < precision):
            print("Precisao %f alcancada:" % (f_x_new - f_x_old))
            return np.array(res)
    print("Iteraccao maxima alcancada")
    return np.array(res)

x0 = -8
res = gradient_descent(x0, f_x, grad_f_x)
plt.plot(res[:,0], res[:, 1], '+')
main_plot()
plt.show()



X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
U, V = np.meshgrid(X, Y)

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V)
ax.quiverkey(q, X=0.3, Y=1.1, U=20,label='Quiver key, length = 10', labelpos='E')

plt.show()

img = np.array([[88, 93, 42, 25, 36, 14, 59, 46, 77, 13, 52, 58],
       [43, 47, 40, 48, 23, 74, 12, 33, 58, 93, 87, 87],
       [54, 75, 79, 21, 15, 44, 51, 68, 28, 94, 78, 48],
       [57, 46, 14, 98, 43, 76, 86, 56, 86, 88, 96, 49],
       [52, 83, 13, 18, 40, 33, 11, 87, 38, 74, 23, 88],
       [81, 28, 86, 89, 16, 28, 66, 67, 80, 23, 95, 98],
       [46, 30, 18, 31, 73, 15, 90, 77, 71, 57, 61, 78],
       [33, 58, 20, 11, 80, 25, 96, 80, 27, 40, 66, 92],
       [13, 59, 77, 53, 91, 16, 47, 79, 33, 78, 25, 66],
       [22, 80, 40, 24, 17, 85, 20, 70, 81, 68, 50, 80]])
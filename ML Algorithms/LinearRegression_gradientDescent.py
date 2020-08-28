import numpy as np 
import matplotlib.pyplot as plt

class LinearRegression:
    '''
        Simple linear regression.

        Parameters :
            X = int/float type (Independent Variable)
            y = int/float type (Dependent Variable)
            learning_rate = float type 
    '''

    def __init__(self , X , y):
        self.X = X
        self.y = y
        self.b = [0,0]


    def update_coef(self , learning_rate):
        y_pred = self.predict()
        y = self.y 
        m = len(y)
        self.b[0] = self.b[0] - (learning_rate * ((1/m) * np.sum(y_pred - y)))
        self.b[1] = self.b[1] - (learning_rate * ((1/m) * np.sum(y_pred -y) * self.X))

    def predict(self , X=[]):
        y_pred = np.array([])
        if not X: X = self.X
        b = self.b

        for x in X:
            y_pred = np.append(y_pred , b[0] + (b[1] * x))

        return y_pred

    def accuracy(self , y_pred):
        p ,e = y_pred , self.y
        n = len(y_pred)

        return 1-sum([
            abs(p[i] -e[i])/e[i]
            for i in range(n)
            if e[i] != 0
        ])

    def compute_cost(self , y_pred):
        m = len(self.y)
        J = (1/2*m) * (np.sum(y_pred - self.y)**2)
        return J

    def plot_best_fit(self, y_pred , fig):
        '''
            Plot best fit line
        '''
        f = plt.figure(fig)
        plt.scatter(self.X , self.y , color= 'k')
        plt.plot(self.X , y_pred , color='g')
        f.show()


def main():
    X  = np.array([i for i in range(11)])
    y = np.array([2*i for i in range(11)])

    reg = LinearRegression(X , y)

    iterations = 0
    steps = 100
    learning_rate = 0.01
    costs = []

    # best fit line
    y_pred = reg.predict()
    reg.plot_best_fit(y_pred , 'Initial Best Fit Line')

    while 1:
        y_pred = reg.predict()
        cost = reg.compute_cost(y_pred)
        costs.append(cost)
        reg.update_coef(learning_rate)

        iterations += 1
        if iterations % steps == 0:
            pritn(iterations , 'Epochs elapsed')
            print('Current accuracy is :', reg.accuracy(y_pred))

            stop = input('Do you to break (y/n) ')
            if stop == 'y':
                break

    # final best fit line
    reg.plot_best_fit(y_pred , 'Final Best fit Line')

    # plot to verify cost function decreases
    h = plt.figure('Verification')
    plt.plot(range(iterations) , costs , color= 'k')
    h.show()

    reg.predict([i for i in range(10)])

if __name__ == "__main__":
    main()

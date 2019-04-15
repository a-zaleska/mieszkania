import pandas
import numpy as np

# Hipoteza (wersja macierzowa)
def hMx(theta, X):
    return X * theta

# Wersja macierzowa funkcji kosztu
def JMx(theta,X,y):
    m = len(y)
    J = 1.0 / (2.0 * m) * ((X * theta - y).T * ( X * theta - y))
    return J.item()

# Wersja macierzowa gradientu funkcji kosztu
def dJMx(theta,X,y):
    return 1.0 / len(y) * (X.T * (X * theta - y))

# Implementacja algorytmu gradientu prostego za pomocą numpy i macierzy
def GDMx(fJ, fdJ, theta, X, y, alpha=0.1, eps=10**-3):
    current_cost = fJ(theta, X, y)
    logs = [[current_cost, theta]]
    while True:
        theta = theta - alpha * fdJ(theta, X, y) # implementacja wzoru
        current_cost, prev_cost = fJ(theta, X, y), current_cost
        if current_cost > 10000:
            break
        if abs(prev_cost - current_cost) <= eps:
            break
        logs.append([current_cost, theta])
    return theta, logs


FEATURES = [
    'Powierzchnia w m2'
 #   'Liczba pokoi',
 #  'Liczba pięter w budynku',
 #   'Piętro',
 #   'Rok budowy'
]

train_data = pandas.read_csv(
    'train/train.tsv',
    header = 0,
    sep = '\t'
    #usecols = ['cena', 'Powierzchnia w m2']
)
X_train = train_data[FEATURES]
Y_train = train_data['cena']
columns = train_data.columns[1:]

m, n = X_train.shape #m - liczba przykładów, n - liczba cech
print(m, n)

# Dodaj kolumnę jedynek do macierzy
XMx = np.matrix(np.concatenate((np.ones((m, 1)), X_train), axis=1)).reshape(m, n + 1)
yMx = np.matrix(Y_train).reshape(m, 1)

thetaStartMx = np.matrix([0, 0]).reshape(2, 1)
best_theta, log = GDMx(JMx, dJMx, thetaStartMx, XMx, yMx, alpha = 0.000075, eps = 0.05)
print(best_theta)

test_data = pandas.read_csv('dev-0/in.tsv', header = None, sep = '\t', names = columns)

X_test = test_data[FEATURES]
m_test, n = X_test.shape
print(m_test, n)

XMx_test = np.matrix(np.concatenate((np.ones((m_test, 1)), X_test), axis=1)).reshape(m_test, n + 1)
print(XMx_test)

YMx_test =  XMx_test*best_theta

#print(YMx_test)

pandas.DataFrame(YMx_test).to_csv('dev-0/out.tsv', index = None, header=None, sep = '\t')

testA_data = pandas.read_csv('test-A/in.tsv', header = None, sep = '\t', names = columns, usecols=['Powierzchnia w m2'])
m_testA, n_testA = testA_data.shape
print(m_testA, n_testA)

XMx_testA = np.matrix(np.concatenate((np.ones((m_testA, 1)), testA_data), axis = 1)).reshape(m_testA, n_testA + 1)
#print(XMx_testA[:10])

Ymx_testA = XMx_testA * best_theta

pandas.DataFrame(Ymx_testA).to_csv('test-A/out.tsv', index = None, header = None, sep = '\t')





print(testA_data[:10])

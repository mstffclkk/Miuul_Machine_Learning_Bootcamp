######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################
df = pd.read_csv("/Users/mstff/CourseProject/Miuul_Python_Machine_Learning/pythonProject/datasets/advertising.csv")
df.shape
df.head()

X = df[["TV"]]              # bağımsız değişken
y = df[["sales"]]           # bağımlı değişken


##########################
# Model
##########################
""" 
y_hat = b + w*TV
y_hat = b + w*X (tek değişkenli linear reg model formulasyonu)
b: sabit (b - bias)
w: ağırlık (teta, w, coefficient)
x: bağımsız değişken
""" 


# model kurma
reg_model = LinearRegression().fit(X, y)

# sabit (b - bias)
reg_model.intercept_[0]             # array olduğu için 0. elemanı seçiyoruz.

# (x) in tv'nin katsayısı (w1)
reg_model.coef_[0][0]


##########################
# Tahmin
##########################
# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?
reg_model.intercept_[0] + reg_model.coef_[0][0]*150


# 500 birimlik tv harcaması olsa ne kadar satış olur?
reg_model.intercept_[0] + reg_model.coef_[0][0]*500

df.describe().T


# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


##########################
# Tahmin Başarısı
##########################
"""
mean_squared_error(y, y_pred)
y: gerçek değer
y_pred: tahmin edilen değer  (reg_model.predict(X))

reg_model.predict(X) --> burada X yukarıda atamış olduğumuz TV sütunu. Diyoruz ki sen bu tv derğerlerini al bana tahmini
                         sales değerlerini ver.( yani bağımsız değikeni al, elimizde yokmuş gibi bağımlı değişkenleri tahmin et)

"""

# MSE
# reg_model yukarıda bizim kurmuş olduğumuz modeldir.
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)           # 10.51
y.mean()
y.std()

# RMSE
np.sqrt(mean_squared_error(y, y_pred))  # 3.24

# MAE
mean_absolute_error(y, y_pred)          # 2.54

# R-KARE (Veri setindeki bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesidir.)
reg_model.score(X, y)                   # 0.612

# Not: değişken sayısı arttıkça r-kare şişmeye meyillidir. düzeltilmiş r-kare değerinin de göz önünde bulundurulması gerekir.
# Not: istatistiki çıktılarla ilgilenmiyoruz.


######################################################
# Multiple Linear Regression
######################################################
df = pd.read_csv("/Users/mstff/CourseProject/Miuul_Python_Machine_Learning/pythonProject/datasets/advertising.csv")

X = df.drop('sales', axis=1)        # bağımlı değişkeni atıp kaydedersek bağımsız değişkenlerin hepsini seçmiş oluruz.

y = df[["sales"]]                   # bağımlı değişken


##########################
# Model
##########################
""" 
train_test_split: train ve test setine ayırmayı sağlar.
                 Bağımlı ve bağımsız değişkenleri alır.
test_size=0.20: test setinin boyutunu %20 yapar.(train seti %80 olmuş olur)

train_test_split bu fonksiyonun çıktısı.

train setinde X, y test setinde X, y verir.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

y_test.shape
y_train.shape

# reg_model = LinearRegression()
# reg_model.fit(X_train, y_train)

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias - intercept)
reg_model.intercept_        # 2.90

# coefficients (w - weights - coef)
reg_model.coef_             # tv: 0.0468431 , radio: 0.17854434, newspaper: 0.00258619


##########################
# Tahmin
##########################
# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90
# 0.0468431 , 0.17854434, 0.00258619

# model denklemi --> Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619 # tahmin edilen satış

# fonksiyonel olarak yazarsak
yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri) # 6.202131

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))    # 1.73

# TRAIN RKARE
reg_model.score(X_train, y_train)               # 0.8959372632325174

# Test RMSE 
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))     # 1.41

# normalde test rmse > train rmse çıkması lazım ama düşük çıkmış güzel bir durum

# Test RKARE
reg_model.score(X_test, y_test)                 # 0.8927605914615384


# cross validation
# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1.69


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71




######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

# Cost function MSE (MSE değerini hesaplar) (güncellenene ağırlıkların hata oranına bakmak)
def cost_function(Y, b, w, X):
    """
    Cost function MSE

    Args:
        Y : bağımlı değişken
        b : sabit, bias
        w : weight, ağırlık
        X : bağımsız değişken

    Returns:
        mse: mean square error
    """
    m = len(Y)      # gözlem sayısı
    sse = 0         # sum of square error

    for i in range(1, m+1):
        y_hat = b + w * X[i]        # tahmin edilen değer
        y = Y[i]                    # gerçek değer
        sse += (y_hat - y) ** 2

    mse = sse / m   # ortalama hata
    return mse


# update_weights (ağırlıkları güncelleme)
def update_weights(Y, b, w, X, learning_rate):
    """
    update_weights

    Args:
        Y : bağımlı değişken
        b : sabit, bias
        w : weight, ağırlık
        X : bağımsız değişken
        learning_rate: öğrenme oranı

    Returns:
        new_b, new_w
    """
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(1, m+1):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    """
    Train fonksiyonu

    Parameters
    ------
        Y : bağımlı değişken
        initial_b (_type_): initial bias value
        initial_w (_type_): initial weight value
        X (_type_): bağımsız değişken
        learning_rate (_type_): öğrenme oranı
        num_iters (_type_): iterasyon sayısı
        
    Returns
    ------
        cost_history, b, w
    """
    # ilk hatanın raporlandığı bölüm
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []       # hataları gözlemleyip saklamak için

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        # her 100 iterasyon da raporla 
        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

    # iterasyon sayısı sonu raporlama
    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

# normal denklemler yöntemiyle gradient descent arasında doğrusal regresyon açısından katsayı bulma ağırlık bulma açısından ne fark var?
# parametre: modelin veriyi kullanarak veriden hareketle bulduğu değerlerdir.(ağırlıklar w, b)
# hiperparametre: veri setinden bulunamayan, kullanıcı tarafından ayarlanması gereken değerlerdir.(initial_b, initial_w, X, learning_rate, num_iters)

df = pd.read_csv("/Users/mstff/PycharmProjects/pythonProject/datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 3000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)


# hatanın düşmediği gözlemlenirse
# learning_rate değeri ile oynanabilir, yeni değişkenler eklenebilir









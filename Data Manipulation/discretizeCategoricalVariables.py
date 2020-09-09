# Imports
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Constructing Dataset
rs = np.random.RandomState(42)
X = rs.uniform(-3,3 , size=100)
y = np.sin(X) + rs.normal(size=len(X)) / 3
X = X.reshape(-1 ,1)

# transform data
encoder = KBinsDiscretizer(n_bins=10 , encode='onehot')
X_binned = encoder.fit_transform(X)

# Predict original dataset
fig , (ax1 , ax2) = plt.subplots(ncols=2 , sharey=True , figsize=(10,4))
line = np.linspace(-3 , 3 ,1000 , endpoint=False).reshape(-1,1)

lr_reg = LinearRegression().fit(X , y)

ax1.plot(line , lr_reg.predict(line) , 
		linewidth=2 , 
		color='g' ,
		label='Linear Regression' )

dt_reg = DecisionTreeRegressor(min_samples_split=3 , random_state=0).fit(X, y)
ax1.plot(line , dt_reg.predict(line) ,
		 linewidth=2 ,
		 color='r' ,
		 label='Decision Tree')
ax1.plot(X[: , 0] , y , 'ok')
ax1.legend(loc='best')
ax1.set_ylabel('Regression Output')
ax1.set_xlabel('Input Feature')
ax1.set_title('Result Before Discretization')

# Predict transformed data
line_binned = encoder.transform(line)

lr_reg = LinearRegression().fit(X_binned , y)

ax2.plot(line , lr_reg.predict(line_binned) , 
		 linewidth=2 , 
		 color='g',
		 linestyle='-',
		 label='Linear Regression')

dt_reg = DecisionTreeRegressor(min_samples_split=3 , random_state=0).fit(X_binned , y)
ax2.plot(line , dt_reg.predict(line_binned) , 
		 linewidth=2 ,
		 color='r',
		 linestyle=':',
		 label='Decision Tree' )
ax2.plot(X[: , 0] , y , 'ok')
ax2.vlines(encoder.bin_edges_[0] , *plt.gca().get_ylim() , linewidth=1 ,alpha=.2)
ax2.legend(loc='best')
ax2.set_xlabel('Input Features')
ax2.set_title('Result After Discretization')

plt.tight_layout()
plt.show()
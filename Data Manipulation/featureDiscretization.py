# Imports
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils._testing import ignore_warning
from sklearn.exceptions import ConvergenceWarning

h = 0.05 # step size in mesh

def get_name(estimator):
	name = estimator.__class__.__name__
	if name == "Pipeline":
		name = [get_name(est[1]) for est in estimator.steps]
		name = " + ".join(name)
	return name

# list of estimator and param_grid , where param_grid is used GridSearchCV
classifiers = [
		(LogisticRegression(random_state=0) ,
		{ "C" : np.logspace(-2,7,10)
			}),
		(LinearSVC(random_state=0) , 
		{ "C" :np.logspace(-2,7,10) 
			}),
		( make_pipeline(
			KBinsDiscretizer(encode='onehot'),
			LogisticRegression(random_state=0) ,
			{ "kbinsdiscretizer_n_bins": np.arange(2,10) ,
			  "logisticregression_C":np.logspace(-2,7,10)}
			  )),
		(make_pipeline(
			KBinsDiscretizer(encode='onehot') , 
			LinearSVC(random_state=0) ,
			{ "kbinsdiscretizer_n_bins": np.arange(2,10) ,
			  "linearsvc_C": np.logspace(-2,7,10)}
			  )),
		(GradientBoostingClassifier(n_estimators=50 , random_state=0) ,
			{ 'learning_rate':	np.logspace(-4,0,10)}
			),
		(SVC(random_state=0) ,
			{'C':np.logspace(-2,7,10)}
			),
]

names = [get_name(e) for e , g in classifiers]

n_samples = 100

datasets=[
		make_moon(n_samples=n_samples ,
				   noise=0.2 , 
				   random_state=0) ,
		make_circles(n_samples=n_samples ,
					 noise=0.2 ,
					 random_state=1) ,
		make_classification(n_samples=n_samples ,
							random_state=2 , 
							n_features=2 ,
							n_redundant=0 ,
							n_informative=2 , 
							n_clusters_per_class=1 )
		]

fig , axes = plt.subplots(nrows=len(datasets) , ncols=len(classifiers) + 1 ,figsize=(21,9))

cm = plt.cm.PiYG
cm_bright= ListedColormap(['#b30036', '#00804d'])

for ds_count , (X , y) in enumerate(datasets):
	print('\n Datasets {} \n'.format(ds_count))

	## Preprocess dataset
	X = StandardScaler().fit_transform(X)
	X_train , y_train , X_test , y_test = train_test_split(X , y ,test_size=0.5, random_state=42)

	## Grid for background colors
	x_min, x_max = X[:, 0].min() - 0.5 , X[:, 0].max() + 0.5
	y_min , y_max = X[:, 1].min() -0.5 , X[:, 1].max() + 0.5

	xx , yy = np.meshgrid(np.arange(x_min , x_max , h) , 
						  np.arange(y_min , y_max , h))

	## Plot datasets first
	ax = axes[ds_count , 0]
	if ds_count==0:
		ax.set_title('Input data')

	## Training data
	ax.scatter(X_train[:, 0] , X_train[:,1] , c= y_train , cmap=cm_bright , edgecolors='k')

	## Test data
	ax.scatter(X_test[:,0] , X_test[:, 1] , c= y_test , cmap=cm_bright , edgecolors='k' , alpha=0.6)
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xticks(())
	ax.set_yticks(())

	## Iterate over Classifiers
	for est_ind , (name , (estimator, param_grid)) in enumerate(zip(names, classifiers)) :
		ax = axes[ds_count , est_ind +1 ]

		clf = GridSearchCV(estimator=estimator , param_grid=param_grid)
		with ignore_warning(category=ConvergenceWarning):
			clf.fit(X_train , y_train)
		score = clf.score(X_test , y_test)
		print('%s: %.2f' % (name, score))

		## plot decision boundary
		if hasattr(clf , "decision_funtion"):
			Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

		else:
			Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
		
		# put the 	result into a color plot
		Z = Z.reshape(xx.shape)
		ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

		# plot the training points
		ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
		           edgecolors='k')
		# and testing points
		ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
		           edgecolors='k', alpha=0.6)
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xticks(())
		ax.set_yticks(())

		if ds_cnt == 0:
		    ax.set_title(name.replace(' + ', '\n'))
		ax.text(0.95, 0.06, ('%.2f' % score).lstrip('0'), size=15,
		        bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'),
		        transform=ax.transAxes, horizontalalignment='right')


plt.tight_layout()

# Add suptitles above the figure
plt.subplots_adjust(top=0.90)
suptitles = [
'Linear classifiers',
'Feature discretization and linear classifiers',
'Non-linear classifiers',
]
for i, suptitle in zip([1, 3, 5], suptitles):
	ax = axes[0, i]
	ax.text(1.05, 1.25, suptitle, transform=ax.transAxes,
        horizontalalignment='center', size='x-large')
plt.show()

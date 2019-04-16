from mdn_helper_functions import * 
import sys

from keras.layers import Dense, Dropout
import numpy
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam,SGD,RMSprop,Adagrad

from keras.callbacks import Callback

from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterSampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_context("notebook", font_scale=1.2)
#sns.set_style("ticks")
import pandas as pd

from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
import json
import h5py
# U82486789	45(1)	38(1)	50(3)	60(2)	40(3)	43(1)	70(1)	54(1)	39(1)	34(1)
# 25, 12, force 25
dataset_id=sys.argv[1]
tindex=int(sys.argv[2])
num_components=int(sys.argv[3])



count_n_splits=10 # KFolds
n_iter=20 #Parameter combinations

dataset = numpy.loadtxt("../data/utias_derived/assessment/dataset-%s/noisy_data_%s.csv"%(dataset_id,dataset_id), delimiter=",",skiprows=1)
num_examples=dataset.shape[0]


### 
## You must use 10-fold cross validation to find appropriate hyper-parameters,
## for your collection of mixture density networks. Remember that all your datasets are not the same and, as a result, 
## your neural network architectures and training can be different amongst the datasets. 
## Some hyper-parameters include, but are not limited to, 
## the number of layers, 
## the number of perceptrons, 
## the learning rate, 
## the optimizing method, 
## the activation function, and 
## whether or not to use regularization and the amount of regularization to use

### Use seven (7) or less layers in your architecture. This includes the mixture layer.
## Use 50 or less units (neurons) in each layer.
## Use tanh, sigmoid, relu, or linear as your activation functions in the layers before the mixture layer.
## Use large mini-batches (from 64 to 512).
## Use learning rates less than 0.1.
###

#for tindex in range(0,2):
xtag="bearing"
if(tindex<1):
	xtag="range"
print "p(%s|x,y)" % xtag
tag=xtag+"_"+dataset_id

print "# of examples: %d" % num_examples
num_test_examples=int(num_examples*0.2)
print "Train: %d, Test: %d" % ((num_examples-num_test_examples),num_test_examples)

numpy.random.seed(10)
numpy.random.shuffle(dataset)
test,train = dataset[:num_test_examples,:], dataset[num_test_examples:,:]


X_train = train[0:,2:]
print "[Training] X:",X_train.shape
Y_train = train[0:,tindex:tindex+1]
print "[Training] Y:",Y_train.shape

X_test = test[0:,2:]
print "[Testing] X:",X_test.shape
Y_test = test[0:,tindex:tindex+1]
print "[Testing] Y:",Y_test.shape

def get_optimizer(optimizer_tag,learing_rate):
	x_optimizer=Adam(lr=learing_rate)
	if(optimizer_tag=='SGD'):
		x_optimizer=SGD(lr=learing_rate)
	if(optimizer_tag=='RMSprop'):
		x_optimize=RMSprop(lr=learing_rate)
	if(optimizer_tag=='Adagrad'):
		x_optimizer=Adagrad(lr=learing_rate)
	return x_optimizer


def get_x_model(Xx_train,activation,no_of_layer,no_of_neuron):#,dropout_prob
	#print Xx_train.shape[1]
	raw_input = Input(shape=(Xx_train.shape[1],), name = 'raw_input')
	x=raw_input
	for i in range(no_of_layer):
		d=Dense(no_of_neuron, activation=activation,name="dense_%d"%i)
		x=d(x)
		#drop=Dropout(dropout_prob,name='dropout_%d'%i)
		#x=drop(x)
	
	mixture=add_univariate_mixture_layer(x,num_components)


	model = Model(inputs=raw_input, outputs=mixture)
	return model



### Grid Search ###
batch_size = [64,128,256,512]#
epochs = [5, 10]#
optimizer = ['SGD', 'Adam']#
no_of_layer = [1,2,3,5]#
no_of_neuron = [8,16,32]#
learing_rate = [0.001, 0.001, 0.0001]#
activation = ['relu', 'tanh', 'sigmoid', 'linear']#
#dropout_prob = [0.0, 0.3, 0.5, 0.8]#

param_dict = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,no_of_layer=no_of_layer,no_of_neuron=no_of_neuron,learing_rate=learing_rate,activation=activation)#dropout_prob=dropout_prob
#print param_dict


kf = KFold(n_splits=count_n_splits)


param_list = list(ParameterSampler(param_dict, n_iter=n_iter))
param_output=[]
min_mse=1.0
optimal_d={}
for i in range(len(param_list)):
	d=param_list[i]
	print "n_iter: %d/%d" % (i+1,n_iter),d
	

	avg_mse=0
	avg_r2s=0

	kf_count=0
	for tx, tsx in kf.split(X_train):
		X_train_kf, X_test_kf = X_train[tx], X_train[tsx]
		Y_train_kf, Y_test_kf = Y_train[tx], Y_train[tsx]


		#print "cross-validation"
		#print X_train_kf.shape,Y_train_kf.shape
		#print X_test_kf.shape,Y_test_kf.shape
		x_model=get_x_model(X_train_kf,d['activation'],d['no_of_layer'],d['no_of_neuron'])# ,d['dropout_prob']
		#print x_model.summary()

		#x_optimizer = Adam(lr=0.001)
		x_model.compile(loss=negative_log_likelihood_loss(num_components), optimizer=get_optimizer(d['optimizer'],d['learing_rate']))

		

		x_history=x_model.fit(X_train_kf, Y_train_kf, epochs=d['epochs'], batch_size=d['batch_size'])

		x_preds=x_model.predict(X_test_kf)
		x_mix_coeff_matrix=x_preds[:,0:num_components]
		#print "MX:",mix_coeff_matrix.shape
		x_means_matrix=x_preds[:,num_components:2*num_components]
		x_stdvs_matrix=x_preds[:,2*num_components:]
		x_total_mean, x_total_var=compute_mixture_total_mean_variance(x_mix_coeff_matrix,x_means_matrix,x_stdvs_matrix)


		xy_trues=Y_test_kf[:,0]
		#print xy_trues[0]
		xy_preds=x_total_mean
		#print xy_preds[0]

		#print len(y_trues),len(y_preds)
		try:
			x_rmse = mean_squared_error(xy_trues, xy_preds)
			print "MSE: ",x_rmse
			avg_mse+=x_rmse
			x_r2s = r2_score(xy_trues, xy_preds)
			print "R2 score: ",x_r2s
			avg_r2s+=x_r2s
			kf_count+=1
		except ValueError:
			continue;
		#break;
		
	try:
		avg_mse=float(avg_mse/kf_count)
		avg_r2s=float(avg_r2s/kf_count)
		d['avg_mse']=avg_mse
		d['avg_r2s']=avg_r2s
		print "MSE: %f" % avg_mse
		print "R2 score: %f" % avg_r2s

		if(avg_mse<min_mse):
			min_mse=avg_mse
			optimal_d=d

		param_output.append(d)
	except ValueError:
		continue;
	#break;
with open('../logs/grid_search_params_%s.json'%tag, 'w') as f:
	json.dump(param_output, f)

with open('../logs/grid_search_params_optimal_%s.json'%tag, 'w') as f:
	json.dump(optimal_d, f)
#####




#### TEST WITH FINE TUNED PARAMS ######
print "optimal params: ",str(optimal_d)
model=get_x_model(X_train,optimal_d['activation'],optimal_d['no_of_layer'],optimal_d['no_of_neuron'])#,optimal_d['dropout_prob']
model.compile(loss=negative_log_likelihood_loss(num_components), optimizer=get_optimizer(optimal_d['optimizer'],optimal_d['learing_rate']))

history=model.fit(X_train, Y_train, epochs=optimal_d['epochs'], batch_size=optimal_d['batch_size'])

preds=model.predict(X_test)
# mix_coeff_matrix=preds[:,0:num_components]
# #print "MX:",mix_coeff_matrix.shape
# means_matrix=preds[:,num_components:2*num_components]
# stdvs_matrix=preds[:,2*num_components:]

mix_coeff_matrix,means_matrix,stdvs_matrix=separate_mixture_matrix_into_parameters(preds,num_components)

total_mean, total_var=compute_mixture_total_mean_variance(mix_coeff_matrix,means_matrix,stdvs_matrix)
max_comp_mean,max_comp_stdv=compute_max_component_mean_variance(mix_coeff_matrix,means_matrix,stdvs_matrix)


trues=Y_test[:,0]
ppreds=total_mean


mse = mean_squared_error(trues, ppreds)
print "MSE: ",mse
r2s = r2_score(trues, ppreds)
print "R2 score: ",r2s


################ SAVE OBJECTS #####################

np.save('../models/X_test_%s.npy'%tag, X_test)
np.save('../models/Y_test_%s.npy'%tag, Y_test)

model.save('../models/model_%s.h5'%tag)

model.save_weights('../models/weights_%s.h5'%tag)

##################################################




#################### PLOTS #######################
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig("../plots/_loss_%s.pdf"%tag)
plt.clf()


## plot params
scale=2
alpha=0.8

## means plot
fig, ax = plt.subplots()
means_df=pd.DataFrame()
means_df['reference(%s)'%xtag]=trues
plt.scatter(means_df['reference(%s)'%xtag], means_df['reference(%s)'%xtag],label='reference(%s)'%xtag,s=scale,alpha=0.8)
for i in range(num_components):
	col_tag='mean(k=%d)'%(i+1)
	means_df[col_tag]=means_matrix[0:,i]
	plt.scatter(means_df['reference(%s)'%xtag], means_df[col_tag],label=col_tag,s=scale,alpha=0.8)
means_df['mean(total)']=total_mean
means_df['mean(max)']=max_comp_mean
plt.scatter(means_df['reference(%s)'%xtag], means_df['mean(total)'],label='mean(total)',s=scale,alpha=0.8)
plt.ylabel('mean')
plt.xlabel(xtag)
ax.legend()
plt.savefig("../plots/_means_%s.pdf"%tag)
plt.clf()

## meax, total mean plot
fig, ax = plt.subplots()
plt.scatter(means_df['reference(%s)'%xtag], means_df['mean(total)'],label='mean(total)',s=scale,alpha=0.8)
plt.scatter(means_df['reference(%s)'%xtag], means_df['mean(max)'],label='mean(max)',s=scale,alpha=0.8)
plt.ylabel('mean')
plt.xlabel(xtag)
ax.legend()
plt.savefig("../plots/_means_max_%s.pdf"%tag)
plt.clf()

## stdvs plot
fig, ax = plt.subplots()
stdvs_df=pd.DataFrame()
stdvs_df['reference(%s)'%xtag]=trues
for i in range(num_components):
	col_tag='stdvs(k=%d)'%(i+1)
	stdvs_df[col_tag]=stdvs_matrix[0:,i]
	plt.scatter(stdvs_df['reference(%s)'%xtag], stdvs_df[col_tag],label=col_tag,s=scale,alpha=0.8)
stdvs_df['stdvs(total)']=total_var
stdvs_df['stdvs(max)']=max_comp_stdv
#print stdvs_df
plt.ylabel('stdv')
plt.xlabel(xtag)
ax.legend()
plt.savefig("../plots/_stdvs_%s.pdf"%tag)
plt.clf()


## meax, total var plot
fig, ax = plt.subplots()
plt.scatter(stdvs_df['reference(%s)'%xtag], stdvs_df['stdvs(total)'],label='stdvs(total)',s=scale,alpha=0.8)
plt.scatter(stdvs_df['reference(%s)'%xtag], stdvs_df['stdvs(max)'],label='stdvs(max)',s=scale,alpha=0.8)
plt.ylabel('var')
plt.xlabel(xtag)
ax.legend()
plt.savefig("../plots/_stdvs_max_%s.pdf"%tag)
plt.clf()

## coeffs plot
fig, ax = plt.subplots()
mix_coeff_df=pd.DataFrame()
mix_coeff_df['reference(%s)'%xtag]=trues
for i in range(num_components):
	col_tag='mix_coeff(k=%d)'%(i+1)
	mix_coeff_df[col_tag]=mix_coeff_matrix[0:,i]
	plt.scatter(mix_coeff_df['reference(%s)'%xtag], mix_coeff_df[col_tag],label=col_tag,s=scale,alpha=0.8)
#print mix_coeff_df
plt.ylabel('mix_coeff')
plt.xlabel(xtag)
ax.legend()
plt.savefig("../plots/_mix_coeff_%s.pdf"%tag)
plt.clf()


	

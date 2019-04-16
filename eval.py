from mdn_helper_functions import * 
import sys

from keras.layers import Dense, Dropout
import numpy as np
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

# U82486789	45(1)	38(1)	50(3)	60(2)	40(3)	43(1)	70(1)	54(1)	39(1)	34(1)
# 25, 12, force 25
dataset_id=sys.argv[1]
#tindex=int(sys.argv[2])
num_components=int(sys.argv[2])
optimal_params_paths=[sys.argv[3],sys.argv[4]]

train = np.loadtxt("../data/utias_derived/assessment/dataset-%s/noisy_data_%s.csv"%(dataset_id,dataset_id), delimiter=",",skiprows=1)
num_train_examples=train.shape[0]
test = np.loadtxt("../data/utias_derived/assessment/dataset-%s/test_noisy_data_%s.csv"%(dataset_id,dataset_id), delimiter=",",skiprows=1)
num_test_examples=test.shape[0]

print "# of training examples: %d" % num_train_examples
print "# of test examples: %d" % num_test_examples

def get_tag(tindex):
	xtag="bearing"
	if(tindex<1):
		xtag="range"
	print "p(%s|x,y)" % xtag
	tag=xtag+"_"+dataset_id+"_v2"
	return xtag,tag

for tindex in [0,1]:

	xtag,tag=get_tag(tindex)


	X_train = train[0:,2:]
	print "[Training] X:",X_train.shape
	Y_train = train[0:,tindex:tindex+1]
	print "[Training] Y:",Y_train.shape

	X_test = test[0:,2:]
	print "[Testing] X:",X_test.shape
	Y_test = test[0:,tindex:tindex+1]
	print "[Testing] Y:",Y_test.shape







# model = load_model(model_path,custom_objects={'negative_log_likelihood_loss': negative_log_likelihood_loss})

# model.load_weights(weights_path)

	optimal_d={}
	with open(optimal_params_paths[tindex]) as f:
	    optimal_d = json.load(f)
	    print(optimal_d)




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

	# np.save('../models/X_test_%s.npy'%tag, X_test)
	# np.save('../models/Y_test_%s.npy'%tag, Y_test)

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
	plt.scatter(means_df['reference(%s)'%xtag], means_df['mean(max)'],label='mean(max)',s=scale,alpha=0.8)
	plt.ylabel('mean')
	plt.xlabel(xtag)
	ax.legend()
	plt.savefig("../plots/_means_%s.pdf"%tag)
	plt.clf()

	# ## meax, total mean plot
	# fig, ax = plt.subplots()
	# plt.scatter(means_df['reference(%s)'%xtag], means_df['mean(total)'],label='mean(total)',s=scale,alpha=0.8)
	# plt.scatter(means_df['reference(%s)'%xtag], means_df['mean(max)'],label='mean(max)',s=scale,alpha=0.8)
	# plt.ylabel('mean')
	# plt.xlabel(xtag)
	# ax.legend()
	# plt.savefig("../plots/_means_max_%s.pdf"%tag)
	# plt.clf()

	## stdvs plot
	fig, ax = plt.subplots()
	stdvs_df=pd.DataFrame()
	ixtag,itag=get_tag(1-tindex)
	stdvs_df['reference(%s)'%ixtag]=test[0:,1-tindex]#trues
	for i in range(num_components):
		col_tag='stdvs(k=%d)'%(i+1)
		stdvs_df[col_tag]=stdvs_matrix[0:,i]
		plt.scatter(stdvs_df['reference(%s)'%ixtag], stdvs_df[col_tag],label=col_tag,s=scale,alpha=0.8)
	stdvs_df['stdvs(total)']=np.sqrt(total_var)
	stdvs_df['stdvs(max)']=max_comp_stdv
	plt.scatter(stdvs_df['reference(%s)'%ixtag], stdvs_df['stdvs(total)'],label='stdvs(total)',s=scale,alpha=0.8)
	plt.scatter(stdvs_df['reference(%s)'%ixtag], stdvs_df['stdvs(max)'],label='stdvs(max)',s=scale,alpha=0.8)
	#print stdvs_df
	plt.ylabel('stdv')
	plt.xlabel(ixtag)
	ax.legend()
	plt.savefig("../plots/_stdvs_%s.pdf"%tag)
	plt.clf()


	# ## meax, total var plot
	# fig, ax = plt.subplots()
	# plt.scatter(stdvs_df['reference(%s)'%xtag], stdvs_df['stdvs(total)'],label='stdvs(total)',s=scale,alpha=0.8)
	# plt.scatter(stdvs_df['reference(%s)'%xtag], stdvs_df['stdvs(max)'],label='stdvs(max)',s=scale,alpha=0.8)
	# plt.ylabel('var')
	# plt.xlabel(xtag)
	# ax.legend()
	# plt.savefig("../plots/_stdvs_max_%s.pdf"%tag)
	# plt.clf()

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


	

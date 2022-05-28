#from django.core.files.uploadedfile import InMemoryUploadedFile
#from sklearn.metrics import precision_recall_fscore_support
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from datasets import load_dataset
#import numpy as np
#import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from Levenshtein import distance as lev
from django.db import models
import pandas as pd
import torchaudio
import torch


class File(models.Model):

	title = models.CharField(max_length = 50)
	filename = models.FileField(blank=False)
	bool_predict = models.CharField(max_length = 50, default="ok_predict")
	

class Predict(models.Model):
	transcription = models.CharField(max_length = 200)
	method_predict = models.CharField(max_length = 500, null = True)
	text_predict = models.CharField(max_length = 500, null = True)
	#prediction = models.CharField(max_length = 50,default = '')


	def asr_transcript(input_file):
	
		#directory2 = "./archive/model3/checkpoint-2400"
		directory2 = "https://huggingface.co/fkHug/model3FromWav2vec/checkpoint-2400"
		vocabDirectory = "https://huggingface.co/fkHug/model3FromWav2vec"
		#vocabDirectory = "./archive/model3"
		path = "./media/"
		processor = Wav2Vec2Processor.from_pretrained(vocabDirectory)
		model = Wav2Vec2ForCTC.from_pretrained(directory2)
		speech_array, sampling_rate = torchaudio.load(path + str(input_file))
		resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
		speech = resampler(speech_array).squeeze().numpy()
		print(input_file)
		
		inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
		with torch.no_grad():
			#logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
			logits = model(inputs.input_values).logits
			predicted_ids = torch.argmax(logits, dim=-1)
			transcription = processor.decode(predicted_ids[0])
			print(transcription)


		return transcription.lower()

	
	def traitment(text):
		x=[]
		x = text.split()
		print(x)
		elt = ['lal','ko','ma','ci','tel','foofu']
		for i in range(1,len(x)):
			if  (x[i]=='ma'and x[i+1]=='ci') | (x[i]=='ma' and x[i+1]=='foofu') | (x[i]=='tel'and x[i+1]=='ma')| (x[i]=='foo'and x[i+1]=='fu'):
				x[0]+= ' '+ x[i]+' '+ x[i+1]
				del(x[i:i+2])
				break
				#x[i+2]+= ' '+ x[i+3]+' '+ x[i+1]
			if x[i]== 'ma' and x[i+1]=='foo' and x[i+2]=='fu':
				x[0]+= ' '+ x[i]+ ' '+ x[i+1]+ ' '+ x[i+2] 
				
				del(x[i:i+3])
				break
			if  x[i]in elt:
				x[0]+= ' '+ x[i]
				del(x[i])
				break
						
		for i in range(len(x)):
			if i >=2:
				x[1]+= ' '+ x[i]
		del(x[2:])
		#x[0] = Predict.search_action(x[0])
		return x

	def search_action(exp): 
		 
		df = pd.read_json("./corpus.json", lines=True)
		categorie = df['expression'].values
		action = df['action'].values
		leiv_distance = []
		for i in range(len(categorie)):
			leiv_distance.append(lev(str(exp),str(categorie[i])))
		#print(leiv_distance)
		min_elt = min(leiv_distance)
		#print(min_elt)
		elt_index = leiv_distance.index(min_elt)
		#print(elt_index)
		elt_action = action[elt_index]
		#print(str(elt_action))
		return str(elt_action)


	""" def classfication_test(test):
		#text = asr_transcript()
		#filename = request.files
		model_path="/home/kineubuntu/prediction_classification/models/model.pkl"
		transformer_path = "/home/kineubuntu/prediction_classification/models/transformer.pkl"
		loaded_model = pickle.load(open(model_path, 'rb'))
		loaded_transformer = pickle.load(open(transformer_path, 'rb'))
		test_features=loaded_transformer.transform([test])
		#get_top_k_predictions(loaded_model,test_features,2)

		# get probabilities instead of predicted labels, since we want to collect top 3
		probs = loaded_model.predict_proba(test_features)
		# GET TOP K PREDICTIONS BY PROB - note these are just index
		best_n = np.argsort(probs, axis=1)[:,-1:]
    
		# GET CATEGORY OF PREDICTIONS
		preds=[[loaded_model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]
    
		# REVERSE CATEGORIES - DESCENDING ORDER OF IMPORTANCE
		preds=[ item[::-1] for item in preds]
		print(preds)
    
		return preds """


from django.db import models
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
from datasets import load_dataset
import librosa
import soundfile as sf
import torch
import os
import html
from pathlib import Path
from django.core.files.uploadedfile import InMemoryUploadedFile





class File(models.Model):

	title = models.CharField(max_length = 50)
	filename = models.FileField(blank=False)
	
	#transcription = models.CharField(max_length = 50)
	#filename = request.files


	def load_processor():
		processor = Wav2Vec2Processor.from_pretrained(directory)
		print("processor loading")
		return processor

	def load_model():
		model = Wav2Vec2ForCTC.from_pretrained(directory)
		print("model loading")
		return model

	def asr_transcript(input_file):
	#tokenizer, model = load_model()
		#processor = load_processor()
		#model = load_model()
		#tokenizer = Wav2Vec2Tokenizer.from_pretrained(directory)
		directory2 = "/home/kineubuntu/Models/modelFromWav2vec"
		directory ="fkHug/modelFromWav2vec"
		path = "/home/kineubuntu/contactDjango/media/"
		processor = Wav2Vec2Processor.from_pretrained(directory)
		model = Wav2Vec2ForCTC.from_pretrained(directory)
		speech_array, sampling_rate = torchaudio.load(path + str(input_file))
		#speech, sampling_rate = sf.read(input_file)
		resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
		speech = resampler(speech_array).squeeze().numpy()
		#input_file = input_file.temporary_file_path()
		print(input_file)
		#speech, sampling_rate = sf.read(input_file)
		#speech = librosa.resample(speech, sampling_rate, 16000)
		

		inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
		with torch.no_grad():
			#logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
			logits = model(inputs.input_values).logits
			predicted_ids = torch.argmax(logits, dim=-1)
			transcription = processor.decode(predicted_ids[0])
			print(transcription)

		return transcription.lower()


  




# from django import forms
# from .models import File

# class UploadFileForm(forms.ModelForm):
#     class Meta:
#         model = File
#         fields = ('filename',)


	


	#function to charge models
		


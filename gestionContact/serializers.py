from rest_framework import serializers
from .models import File,Predict

class FileSerializers(serializers.ModelSerializer):
	class Meta():
		model = File
		fields = ('filename','bool_predict',)
		#fields = ('transcription',)


	#def __str__(self):
	#	return {self.transcription, self.prediction}


class PredictSerializers(serializers.ModelSerializer):
	class Meta():
		model = Predict
		fields = ('transcription','method_predict','text_predict')

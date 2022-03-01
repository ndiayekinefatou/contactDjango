from rest_framework import serializers
from .models import File

class FileSerializers(serializers.ModelSerializer):
	class Meta():
		model = File
		fields = ('filename',)
		#fields = ('transcription',)


	#def __str__(self):
	#	return f"{self.transcription}"

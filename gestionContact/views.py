#from django.shortcuts import render
#from django.http import HttpResponse as Response
#from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
#from django.core.files.storage import FileSystemStorage
#from django.urls import reverse_lazy
from .models import File
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser,FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializers
from django.core.files.base import ContentFile
#from .forms import UploadFileForm
import requests
import json
from .apps import GestioncontactConfig
import torchaudio
from datasets import load_dataset
import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy
from pathlib import Path
import io
from django.core.files.uploadedfile import InMemoryUploadedFile



# Create your views here.
#def index(request):
 #   return HttpReponse("Bienvenue dans mon bloc")


# class Home(TemplateView):
#     template_name = 'home.html'


# def upload(request):
#     context = {}
#     if request.method == 'POST':
#         uploaded_file = request.FILES['document']
#         fs = FileSystemStorage()
#         name = fs.save(uploaded_file.name, uploaded_file)
#         context['url'] = fs.url(name)
#     return render(request, 'upload.html', context)

class Home(APIView):
    #template_name = 'home.html'
    parser_classes = (MultiPartParser,FormParser)


    # def post(self,request,*args,**kwargs):
    #     #if request == POST:

    #     file_serializer = FileSerializers(data=request.data)
    #     if file_serializer.is_valid():
    #         file_serializer.save()
    #         return Response(file_serializer.data, status = status.HTTP_201_CREATED)
    #     else:
    #         return Response(file_serializer.errors, status = status.HTTP_400_BAD_REQUEST)


    def post(self,request,*args,**kwargs):
        #if request == POST:
        #if request.method == 'POST':

        print(request.data)
        #if request.fields['filename']:
        file_serializer = FileSerializers(data = request.data)
        #print(request.data)


        #filename = (requests.get('http://10.153.5.124:8000/'))
        if file_serializer.is_valid():
        
            #File.fileField = filename
            #filename = file_serializer.get('file')
            
            file_serializer.save()
            file = request.FILES['filename']
            #files = {'audio_example': file}
            File.asr_transcript(file)
            return Response(file_serializer.data, status = status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status = status.HTTP_400_BAD_REQUEST)


    


    # def post(request,*args,**kwargs):
    #     #if request.method == 'POST':
    #     form = UploadFileForm(request.FILES)
    #     if form.is_valid():
    #         form.save()
    #         return HttpResponse('The file is saved')
    #     else:
    #         form = UploadFileForm()
    #         context = {
    #             'form':form,
    #         }
    #     return render(request, 'upload.html', context)


    # def uploadFile(request):
    #     #Response response;
    #     if request.FILES['filename']:
    #         filename = reponse.get('filename')
    #         model.FileField = filename
    #         model.save()
    #         print(filename)
            #return Response(file_serializer.data, status = status.HTTP_201_CREATED)
        #else:
         #   return Response(file_serializer.errors, status = status.HTTP_400_BAD_REQUEST)

    # def fetchFileFromFlutter(request):
    #     if request.method == 'POST':
    #         #filename = request.files['filename']
    #         filename = json.loads(request.body)
    #         return HttpResponse(filename, content_type='multipart/form-data')
    
    #     else: 
    #         return Response(filename.errors, status = status.HTTP_400_BAD_REQUEST)



    #def post(sel, re)




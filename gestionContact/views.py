#from django.views.generic import TemplateView, ListView, CreateView
#from django.core.files.uploadedfile import InMemoryUploadedFile
#from django.core.files.storage import FileSystemStorage
#from django.http import HttpResponse as Response
#from django.core.files.base import ContentFile
#from .forms import UploadFileForm
#from django.conf import settings
#from fileinput import filename
#import requests


from rest_framework.parsers import MultiPartParser,FormParser
from .serializers import FileSerializers,PredictSerializers
from django.core.files.storage import default_storage
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import Predict




class Home(APIView):
    #template_name = 'home.html'
    parser_classes = (MultiPartParser,FormParser)


  
 
    def post(self,request,*args,**kwargs):
        file_serializer = FileSerializers(data = request.data)
        print(request.data)
        if file_serializer.is_valid():
            file_serializer.save()
          
            file = request.FILES['filename'] 
            #print(request.FILES)
            files = {'audio_example': file} 
            trans = Predict.asr_transcript(file)
            request.data['transcription'] =trans
            transcription = request.data['transcription']
            val_pred = request.data['bool_predict']
            default_storage.delete('audio_example.wav') 
            if(val_pred == 'ok_predict'):
                x = Predict.traitment(transcription)
                x[0] = Predict.search_action(x[0])
                print(x)
                request.data['method_predict'] = x[0]
                request.data ['text_predict'] = x[1]
                print("aa")
                print(request.data)
                serializers= PredictSerializers(data= request.data)
                if serializers.is_valid():

                    return Response(serializers.data, status = status.HTTP_201_CREATED)
                else:
                    return Response(serializers.errors, status = status.HTTP_400_BAD_REQUEST)

            else:
                #pred = ''
                serializers= PredictSerializers(data= request.data)
                if serializers.is_valid():
                    print(serializers.data)
                    return Response(serializers.data, status = status.HTTP_201_CREATED)
                else:
                    return Response(serializers.errors, status = status.HTTP_400_BAD_REQUEST)     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

    # def get(self, requests,*args,**kwargs):
        
    #     pred = Predict.objects.all()
    #     serializer = PredictSerializers(pred, many=True)
    #     print(serializer.data)
    #     return Response(serializer.data)
     
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

    # def post(self,request,*args,**kwargs):
    #     #if request == POST:

    #     file_serializer = FileSerializers(data=request.data)
    #     if file_serializer.is_valid():
    #         file_serializer.save()
    #         return Response(file_serializer.data, status = status.HTTP_201_CREATED)
    #     else:
    #         return Response(file_serializer.errors, status = status.HTTP_400_BAD_REQUEST)


   
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



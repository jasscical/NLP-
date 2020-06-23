from django.http import HttpResponse
from django.shortcuts import render

from index.predict import predict, train
from index.predictChinese import predict_pb
#from util2 import load_yaml_config, texts_to_sequences

# Create your views here.
# def index(request):
#    return HttpResponse("This is index")
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def index(request):
    if request.method == 'POST':

        text = request.POST.get('text')
        print('text', text)
        if(is_Chinese(text)):
            results = predict_pb(text)
        else:
            results = predict(text)
        print(results)
        results['input'] = text
        results['input_length'] = len(text.split())
        return render(request, 'index/index.html', results)

    return render(request, 'index/index.html')


def train_feedback(request):
    text = request.GET.get('text', default='')
    feedback = request.GET.get('feedback')
    print('text', text)
    print('feedback', feedback)
    train(text, feedback)
    return render(request, 'index/index.html')

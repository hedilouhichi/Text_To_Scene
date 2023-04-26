from utils import draw_parallelogram, draw_rhombus, drawrect, metric_conversion, draw_circle, replace_numbers_with_digits_ar, replace_numbers_with_digits_en

import torch.nn as nn
import transformers
import langid
import requests
import stanza
from quantulum3 import parser

class BertForMathProblemClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(BertForMathProblemClassification, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def detect_language(text):
    lang, confidence = langid.classify(text)
    return lang

action_verbs =  ['Clean', 'Drink', 'Play', 'Turn', 'Sit down', 'Smell', 'Ski', 'Wonder', 'Explain', 'Increase', 'Repeat', 'Bathe', 'Run', 'Tell', 'Hug', 'Sit', 'Plan', 'Wash', 'Start', 'Climb', 'Touch', 'Cook', 'Agree', 'Offer', 'Answer', 'Stand', 'Point', 'Check', 'Receive', 'Collect', 'Stand up', 'Ask', 'Enter', 'Continue', 'Rise', 'Leave', 'Enjoy', 'Dream', 'Paint', 'Shake', 'Learn',  'Carry', 'Follow', 'Speak', 'Write', 'Eat', 'Jump', 'Hold', 'Drive', 'Show', 'Use', 'Finish', 'Move', 'Watch', 'Draw', 'Regard', 'Improve', 'Allow', 'Smile', 'Bow', 'Love', 'Dance', 'Hope', 'Meet', 'Choose', 'Grow', 'Take', 'Walk', 'Open', 'Give', 'Reply', 'Exit', 'Travel', 'Change', 'Think', 'Ride', 'Return', 'Like', 'Close', 'Become', 'Create', 'Send', 'Laugh', 'Cry', 'Hear', 'Help', 'Call', 'Find', 'Save', 'Contribute', 'Prepare', 'Begin', 'Solve', 'Study', 'Join', 'Complete', 'Read', 'Act', 'Catch', 'Hide', 'Sell', 'Talk', 'Want']
action_verbs = [word.lower() for word in action_verbs]


def image_generation(seed):
    stanza.download('en')
    
    # This sets up a default neural pipeline in Lang
    print(seed)
    sentences = seed.split('.')
    if (seed.count('.') >= 2) or ((seed.count('.') < 2)and(seed.count('?')>0)):
        deleted_sent = sentences.pop(-1)
        seed = '.'.join(sentences)
    lang=detect_language(seed)
    if lang=='ar':
        seed=replace_numbers_with_digits_ar(seed)
    else:
        seed=replace_numbers_with_digits_en(seed)
    seed1=seed
    print(lang)
    if (lang != 'en'):
        response = requests.get('https://api.mymemory.translated.net/get?q='+seed+'&langpair='+lang+'|en')
        seed1  = response.json()['responseData']['translatedText']
    if ',' in seed:
        seed=seed.replace(',',' , ')
    nlp = stanza.Pipeline(lang, use_gpu=False,
                          processors='tokenize,pos,lemma')
    doc = nlp(seed)
    print(doc)
    res = {'type': None, 'data': []}
    '''
    # Load the trained model and tokenizer
    model = BertForMathProblemClassification()
    model.load_state_dict(torch.load('C:/Users/Asus/Downloads/bert_math_problem_classification.pt'))
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Example math problem
    problem = seed

    # Tokenize the input and convert to tensors
    input_ids = torch.tensor(tokenizer.encode(seed1, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    attention_mask = torch.ones_like(input_ids)

    # Pass the input to the model and get the predicted class
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_class = torch.argmax(outputs).item()

    # Print the predicted class
    class_names = ['Not Geometry', 'Geometry']
    problem_type = class_names[predicted_class]
    '''
    problem = seed
    metrics=[] 
    problem_type='Not Geometry'
    print(problem_type)
    if (problem_type=='Geometry'):      
        language, _ = langid.classify(problem)
        if language != 'en':
            response = requests.get('https://api.mymemory.translated.net/get?q='+problem+'&langpair='+'ar'+'|en')
            translated = response.json()['responseData']['translatedText']
        else:
            translated=problem
        translated_doc=nlp(translated)
        for sent in translated_doc.sentences:
            quants = parser.parse(sent.text)
            for q in quants:
                if q.unit.entity.name == 'length' :
                    print(metrics)
                    metrics.append([float(q.surface.split()[0]), q.unit.uri])
        for i in range(len(metrics)):
            metrics[i]=metric_conversion(metrics[i])    
        print(metrics)        
        if len(metrics) > 0:
            if len(metrics)==1:
                if 'diameter' in problem:
                    #res['type'] = 'diametre'
                    Output_List=[draw_circle((metrics[0][0])/2,str(int(metrics[0][1]))+metrics[0][2])]
                elif 'radius' in problem:
                    #res['type'] = 'radius'
                    Output_List=[draw_circle(metrics[0][0],str(int(metrics[0][1]))+metrics[0][2])]
                else:
                    #res['type'] = 'square'
                    Output_List=[drawrect(metrics[0][0],metrics[0][0],str(int(metrics[0][1]))+metrics[0][2],str(int(metrics[0][1]))+metrics[0][2])]

            elif len(metrics)==2:
                if 'parallelogram' in problem:
                    #res['type'] = 'parallelogram'
                    if(metrics[0][1]==max(metrics[0][1],metrics[1][1])):
                        height=str(int(metrics[0][1]))+str(metrics[0][2])
                        width=str(int(metrics[1][1]))+str(metrics[1][2])
                        r_height=metrics[0][1]
                        r_width=metrics[1][1]
                        c_height=metrics[0][0]
                        c_width=metrics[1][0]
                    else:
                        height=str(int(metrics[0][1]))+str(metrics[0][2])
                        width=str(int(metrics[1][1]))+str(metrics[1][2])
                        r_width=metrics[0][1]
                        r_height=metrics[1][1]
                        c_width=metrics[0][0]
                        c_height=metrics[1][0]
                    Output_List=[draw_parallelogram(c_height,c_width,height,width)]
                elif 'rhombus' in problem:
                    #res['type'] = 'rhombus'
                    if(metrics[0][1]==max(metrics[0][1],metrics[1][1])):
                        height=str(int(metrics[0][1]))+str(metrics[0][2])
                        width=str(int(metrics[1][1]))+str(metrics[1][2])
                        r_height=metrics[0][1]
                        r_width=metrics[1][1]
                        c_height=metrics[0][0]
                        c_width=metrics[1][0]
                    else:
                        height=str(int(metrics[0][1]))+str(metrics[0][2])
                        width=str(int(metrics[1][1]))+str(metrics[1][2])
                        r_width=metrics[0][1]
                        r_height=metrics[1][1]
                        c_width=metrics[0][0]
                        c_height=metrics[1][0]
                    Output_List=[draw_rhombus(c_height,c_width,height,width)]
                else:
                    #res['type'] = 'rectangle'
                    if(metrics[0][1]==max(metrics[0][1],metrics[1][1])):
                        print(str(metrics[0][1]))
                        height=str(int(metrics[0][1]))+str(metrics[0][2])
                        width=str(int(metrics[1][1]))+str(metrics[1][2])
                        r_height=metrics[0][1]
                        r_width=metrics[1][1]
                        c_height=metrics[0][0]
                        c_width=metrics[1][0]
                    else:
                        print(str(int(metrics[0][1])))
                        height=str(int(metrics[0][1]))+str(metrics[0][2])
                        width=str(int(metrics[1][1]))+str(metrics[1][2])
                        r_width=metrics[0][1]
                        r_height=metrics[1][1]
                        c_width=metrics[0][0]
                        c_height=metrics[1][0]
                    print(c_height,c_width,height,width)
                    Output_List=[drawrect(c_height,c_width,height,width)]
            elif len(metrics)==4:
                if 'trapezium' in problem:
                    res['type'] = 'trapezium'
            del doc
            print(len(metrics))
            print(Output_List)
            return Output_List

    else:
        res['type'] = 'entity'
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    for sent in doc.sentences:
        for word in sent.words:
            res['data'].append([word.lemma, word.upos, word.text])

    i = 0
    #for w in res['data']:
     #   w.append(i)
      #  i = 0
       # if w[1] == 'NUM':
        #    print(w[0])
         #   i = int(w[0])
    print(res)
    for i, w in enumerate(res['data']):
        if w[1] == 'NOUN' :
            if i < len(res['data'])-1:
                if res['data'][i+1][1] == 'NOUN':
                    w[0] = w[0]+' '+res['data'][i+1][0]
                    w[2] = w[2]+' '+res['data'][i+1][2]
                    del res['data'][i+1]
    print(res)
    for i, w in enumerate(res['data']):
        if w[1] == 'PROPN':
            if i < len(res['data'])-1:
                if res['data'][i+1][1] == 'PROPN':
                    w[0] = w[0]+' '+res['data'][i+1][0]
                    w[2] = w[2]+' '+res['data'][i+1][2]
                    del res['data'][i+1]
    
    for i, w in enumerate(res['data']):
        if w[1] == 'ADV':
            if (i < len(res['data'])-1) and (i>0):
                print("aaa")
                if (res['data'][i+1][1] == 'VERB') and (res['data'][i+1][0] in action_verbs):
                    w[0] = w[0]+' '+res['data'][i+1][0]
                    w[2] = w[2]+' '+res['data'][i+1][2]
                    del res['data'][i+1]    
                elif (res['data'][i-1][1] == 'VERB') and (res['data'][i-1][0] in action_verbs):
                    w[0] = res['data'][i-1][0] + ' ' + w[0]
                    w[2] = res['data'][i-1][2] + ' ' + w[2]
                    del res['data'][i-1]   

    for i, w in enumerate(res['data']):
        if w[1] == 'X':
            if (i < len(res['data'])-1) and (i>0):
                print("aaa")
                if (res['data'][i+1][1] == 'NOUN') :
                    w[0] = w[0]+' '+res['data'][i+1][0]
                    w[2] = w[2]+' '+res['data'][i+1][2]
                    w[1]='NOUN'
                    del res['data'][i+1]    
                elif (res['data'][i-1][1] == 'NOUN') :
                    w[0] = res['data'][i-1][0] + ' ' + w[0]
                    w[2] = res['data'][i-1][2] + ' ' + w[2]
                    w[1]="NOUN"
                    del res['data'][i-1]   
    print(res)
    dim_numbers=[]
    language, _ = langid.classify(w[0])
    if language != 'en':
        response = requests.get('https://api.mymemory.translated.net/get?q='+problem+'&langpair='+'ar'+'|en')
        translated = response.json()['responseData']['translatedText']
    else:
        translated=problem
    translated_doc=nlp(translated)
    for sent in translated_doc.sentences:
        sent=sent.text.replace(',','and')
        quants = parser.parse(sent)
        print(quants)
        for q in quants:  
            if q.unit.entity.name != 'dimensionless' :   
                dim_numbers.append(q.value)
    print(dim_numbers)
    for w in res['data']:
        if w[1] == 'NOUN':
            language, _ = langid.classify(w[0])
            if language != 'en':
                response = requests.get('https://api.mymemory.translated.net/get?q='+w[0]+'&langpair='+'ar'+'|en')
                translated = response.json()['responseData']['translatedText']
            else:
                translated=w[0]
            url = "https://api.giphy.com/v1/stickers/search?api_key=iidRVNv0y0mmMUNhYrwlVFufRdIeFLJP&q=" + \
                translated+"&limit=1&offset=1&rating=PG"
            print(url)
            response = requests.get(url)
            if (response.json()['data']):
                w[0] = response.json()['data'][0]['images']['downsized']['url']
            else:w[1]=' '
        if w[1] == 'PROPN': #or w[1] == 'X':
            r2 = requests.get("https://api.genderize.io?name="+w[0])
            gender = r2.json()['gender']
            if gender == 'female':
                w[0] = 'https://media.giphy.com/media/ifMNaJBQEJPDuUxF6n/giphy.gif'
            else:
                w[0] = 'https://media.giphy.com/media/TiC9sYLY9nilNnwMLq/giphy.gif'
        if w[1] == 'ADV':
            language, _ = langid.classify(w[0])
            if language != 'en':
                response = requests.get('https://api.mymemory.translated.net/get?q='+w[0]+'&langpair='+'ar'+'|en')
                translated = response.json()['responseData']['translatedText']
            else:
                translated=w[0]
            url = "https://api.giphy.com/v1/stickers/search?api_key=iidRVNv0y0mmMUNhYrwlVFufRdIeFLJP&q=" + \
                translated+"&limit=1&offset=1&rating=PG"  #W[2] better 
            response = requests.get(url)
            if (response.json()['data']):
                w[0] = response.json()['data'][0]['images']['downsized']['url']

        if (w[1] == 'VERB') and (w[0] in action_verbs):
            url = "https://api.giphy.com/v1/stickers/search?api_key=iidRVNv0y0mmMUNhYrwlVFufRdIeFLJP&q=" + \
                w[0]+"&limit=1&offset=1&rating=PG"
            response = requests.get(url)
            if (response.json()['data']):
                w[0] = response.json()['data'][0]['images']['downsized']['url']
        

    Output_List=[]
    for w in res['data']:
        if (w[1]=='NUM') and (not (int(w[0]) in dim_numbers)) and (int(w[0])<15):
            Output_List.append([w[2],1])
            continue
        if (w[0].startswith('https')):
            Output_List.append([w[0],0])
        else:
            Output_List.append([w[2],0])
   #print(res)
    #print("aaaa")
    print(Output_List)
    del doc
    
    return Output_List


from flask import Flask,request,jsonify
import BertForMathProblemClassification as M

app=Flask(__name__)

@app.route('/')
def index():
    return "Hello world"



@app.route('/predict',methods=['POST'])

def predict():
    problem=request.form.get('problem')
    # Appel de votre API pour obtenir le résultat du problème mathématique
    list_output=M.image_generation(problem)
    for i in range(len (list_output)):
        if list_output[i][1]==1:
            for j in range (int(list_output[i][0])-1):
                list_output.insert(i+1,[list_output[i+1][0],0])
    for i in range(len (list_output)):
      if "http" in list_output[i][0]:
        list_output[i][1]=2
    
            
    result = list_output
    print(list_output)
    return jsonify({'images': result})



if __name__=='__main__':
    app.debug = True

    app.run()

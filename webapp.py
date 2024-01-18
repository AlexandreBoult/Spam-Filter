from flask import Flask,render_template,request
import os
import pandas as pd
import Spam_filter_main as sf
import ast
print(os.getcwd())

app = Flask(__name__)

df=pd.read_table("SMSSpamCollection")

def reset_stgs():
    global dflt_par,score,ratio,reset
    dflt_par="'n_components':200,   'n_iter':5,   'random_state':1,   'max_iter':400,   'hidden_layer_sizes':(90, 60),   'alpha':0.0056"
    score = 0
    ratio = 0.8
    reset = 0

reset_stgs()

@app.route("/", methods=['GET'])
def index():
    global model,score,dflt_par,ratio,reset
    reset = request.args.get('reset')
    msg = request.args.get('msg')
    trained = request.args.get('train')
    submit = request.args.get('submit')
    par = request.args.get('par')
    if request.args.get('ratio') !=None : ratio = float(request.args.get('ratio'))/20
    cat=None
    if request.method == 'GET':
        if score == "" or score == None : score=0
        if par == "" or par == None : par=dflt_par
        if trained == None : 
            trained=0
        elif trained == "1":
            par_dict=dict(ast.literal_eval("{"+dflt_par+"}"),**ast.literal_eval("{"+par+"}"))
            print("Training model with following parameters :\n" + str(par_dict)[1:-1])
            model,score=sf.train_model(0,ratio,df,par_dict)
            score=round(score,5)
            trained=2
        if submit != None :
            trained=2
            cat=sf.test_msg(model,msg)
        if reset == "1" : reset_stgs()
        return render_template("index.html",trained=trained,msg=msg,cat=cat,par=par,dflt_par=dflt_par,score=score,ratio=ratio)
    return render_template("index.html")


if __name__ == "__main__":
    app.run()

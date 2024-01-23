from imports2 import *


df=pd.read_table("SMSSpamCollection",header=None)
df.columns=["cat","msg"]

def reset_stgs():
    global dflt_par
    dflt_par="'model_name'= 'SVC', 'tsvd__n_components'= 200, 'tsvd__n_iter'= 5, 'tsvd__random_state'= 2, 'svmsmote__random_state'= 2, 'svmsmote__sampling_strategy'= 0.5, 'svmsmote__off'= False, 'svc__random_state'= 2"
    par = dflt_par
    score = 0
    ratio = 0.8
    reset = None
    opti = None
    export = None
    import0 = None
    trained = None
    cat= None
    submit = None
    image = None
    colors = [(44, 46, 63),(74, 77, 105),(99, 0, 192),(124, 0, 240)]
    msg = ""
    return score,ratio,reset,par,export,import0,trained,cat,image,colors,msg


def create_settings(par,id_vault):
    clear_temp(id_vault)
    os.mkdir(f"settings/{id_vault}")
    file=open(f"settings/{id_vault}/settings.json","w")
    file.write("{"+par+"}")
    file.close()


def create_plot(cm):
    colors=reset_stgs()[-2]
    fig = plt.figure(facecolor=[e/255 for e in colors[0]])
    fig,ax=plot_confusion_matrix(conf_mat=cm,figure=fig)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.set_xlabel('predicted label (ham=0, spam=1)')
    for e in ['left','right','bottom','top'] : ax.spines[e].set_color('white')
    ax.tick_params(colors='white')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    image = base64.b64encode(buf.getbuffer()).decode("ascii")
    return image

def clear_temp(id_vault):
    try : 
        shutil.rmtree(f"settings/{id_vault}")
    except : pass

try : os.mkdir(f"settings")
except : pass
for e in os.listdir("settings") : clear_temp(e)

reset_stgs()
model_instances={}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'settings'

@app.route("/", methods=['POST','GET'])
def index():
    global dflt_par,model_instances

    if request.method == 'GET':
        score,ratio,reset,par,export,import0,trained,cat,image,colors,msg=reset_stgs()
        id_vault="".join([str(rd.randint(0,9)) for e in range(100)])
        vault=(score,ratio,reset,par,export,import0,trained,cat,image,colors,msg,id_vault)

    elif request.method == 'POST':
        score,ratio,reset,par,export,import0,trained,cat,image,colors,msg,id_vault=ast.literal_eval(request.form.get('vault'))
        clear_temp(id_vault)

        export = request.form.get('export')
        import0 = request.form.get('import0')
        trained = request.form.get('train')
        reset = request.form.get('reset')
        msg = request.form.get('msg')
        submit = request.form.get('submit')
        par = request.form.get('par')
        opti = request.form.get('opti')
        if request.form.get('ratio') != None : ratio = float(request.form.get('ratio'))/20
        if score == "" or score == None : score=0
        if par == "" or par == None : par=dflt_par
        par_dict=dict(ast.literal_eval("{"+dflt_par.replace("=",":")+"}"),**ast.literal_eval("{"+par.replace("=",":")+"}"))

        if import0 == "1" :
            file = request.files.get('file')
            par=str(file.read())[3:-2]

        if submit == "1" :
            preproc,model,score,cm=model_instances[id_vault]
            trained="2"
            cat={1:'spam',0:'ham'}[test_msg(preproc,model,msg)]

        if trained == "1":
            print("Training model with following parameters :\n" + str(par_dict)[1:-1])
            preproc,model,score,cm=train_model(0,ratio,df,par_dict)
            score=round(score,5)
            model_instances[id_vault]=(preproc,model,score,cm)
            trained="2"
            image=create_plot(cm)

        if opti != None:
            par=train_model(1,ratio,df,par_dict)
            par_dict=dict(ast.literal_eval("{"+dflt_par.replace("=",":")+"}"),**par)
            par=str(par_dict)[1:-1]
            cat=None

        if reset == "1" : score,ratio,reset,par,export,import0,trained,cat,image,colors,msg=reset_stgs()

        if export == "1":
            print("export")
            create_settings(par,id_vault)
            upload_folder = app.config['UPLOAD_FOLDER']
            file_path = os.path.join(current_app.root_path, upload_folder, "settings.json")
            return send_from_directory(directory=upload_folder, path=f"{id_vault}/settings.json", as_attachment=True)

        vault=(score,ratio,reset,par,export,import0,trained,cat,image,colors,msg,id_vault)
        return render_template("index.html",vault=vault)
    
    return render_template("index.html",vault=vault)

if __name__ == "__main__":
    app.run(threaded=True)
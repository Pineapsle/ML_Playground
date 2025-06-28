# Flask aoo for a web UI to interact with the model

from flask import Flask, render_template

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():    
    return render_template('about.html')

# Models Routes

@app.route('/lin_reg', methods=['GET', 'POST'])
def lin_reg():
    # model results
    mse = 1595610795.60622
    rmse = 39945.09727
    r2 = 0.79823
    r = 0.89343
    return render_template(
        'lin_reg.html',
        mse=mse,
        rmse=rmse,
        r2=r2,
        r=r
        )

@app.route('/log_reg', methods=['GET', 'POST'])
def log_reg():
    # model results
    confusion_matrix = [[92, 9,
                        7, 116]]
    precision = 0.92800
    recall = 0.94309
    f1_score = 0.93548
    accuracy = 0.92857
    return render_template(
        'log_reg.html',
        confusion_matrix=confusion_matrix,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        accuracy=accuracy
        )

@app.route('/dec_tree', methods=['GET', 'POST'])
def dec_tree():
    # Model Results
    confusion_matrix = [[20705, 1919,
                        25215, 2337]]
    precision = 0.54911
    recall = 0.08482
    f1_score = 0.14694
    accuracy = 0.45922
    # Iterations
    iterations = {
        0 : 0.6930,
        100 : 0.6928,
        200 : 0.6926,
        300 : 0.6923,
        400 : 0.6919,
        500 : 0.6913,
        600 : 0.6902,
        700 : 0.6885,
        800 : 0.6857,
        900 : 0.6811 
    } 
    return render_template(
        'dec_tree.html',
        confusion_matrix=confusion_matrix,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        accuracy=accuracy,
        iterations=iterations
        )

@app.route('/neural', methods=['GET', 'POST'])
def neural():
    # Model Results
    confusion_matrix = [[20705, 1919,
                        25215, 2337]]
    precision = 0.54911
    recall = 0.08482
    f1_score = 0.14694
    accuracy = 0.45922
    return render_template(
        'neural.html',
        confusion_matrix=confusion_matrix,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        accuracy=accuracy
        )

if __name__ == '__main__':
    app.run(debug=True)
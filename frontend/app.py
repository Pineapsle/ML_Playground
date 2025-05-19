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
    return render_template('lin_reg.html')

@app.route('/log_reg', methods=['GET', 'POST'])
def log_reg():
    return render_template('log_reg.html')

@app.route('/dec_tree', methods=['GET', 'POST'])
def dec_tree():
    return render_template('dec_tree.html')

@app.route('/neural', methods=['GET', 'POST'])
def neural():
    return render_template('neural.html')

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask
from flask import render_template

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/html_default')
def html_default():
    name = ['Alex1', 'Alex2', 'Alex3']
    return render_template('hello.html', name=name)


@app.route('/cover')
def html_cover():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

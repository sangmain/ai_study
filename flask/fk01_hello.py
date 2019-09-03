from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "<h1>world</hi>"

@app.route('/bit/')
def hell33o():
    return "<hi>bit world</hi>"
if __name__ == '__main__':
    app.run(host="192.168.0.180", port=8888, debug=False)
    # app.run(host="127.0.0.1", port=5000, debug=False)

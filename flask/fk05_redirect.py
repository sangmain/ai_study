from flask import Flask
from flask import redirect
app = Flask(__name__)

from flask import make_response
@app.route('/aa/')
def hello():
    return "<h1>redirecting to 'http://www.naver.com'</h1>"

@app.route('/aaaa/')
def index():
    return redirect('http://www.naver.com')

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)
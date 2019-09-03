from flask import Flask, render_template
app = Flask(__name__)

@app.route('/user/<name>')
def user(name):
    return render_template('user2.html', name=name)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

'''
capitalize : 값의 첫번쨰 문자를 대문자로 변환 나머지는 소문자로
lower      : 소문자로 만든다
upper      : 대문자로 만든다
title      : 각 단어의 앞을 capitalize
trim       : 앞부분과 뒷부분에서 공백문자를 제거한다
'''
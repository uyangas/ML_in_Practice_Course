from flask import Flask

app = Flask('test')

@app.route('/myname', methods=['GET'])
def name():
    return "Uyanga"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
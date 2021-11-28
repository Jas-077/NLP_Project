from flask import Flask,render_template, request
import lyrics as lyr

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods = ["POST"] )
def lyrics():
    if request.method == "POST":
        no_words = request.form["no"]
        start_lyr = request.form["lyrics"]
        print(no_words)
        print(start_lyr)
        
        pred = lyr.pre(start_lyr, no_words)
        print(pred)
        return render_template("generate.html", pred = pred, orig = start_lyr)


if __name__ == "__main__":
    app.run()
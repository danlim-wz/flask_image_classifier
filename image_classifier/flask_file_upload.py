from flask import Flask, request, make_response, render_template, redirect

app = Flask(__name__)

@app.route("/upload",methods=["GET","POST"])
def upload_file():
    if request.method == "POST":
        image = request.files["input_file"]
        
    return render_template("upload_button.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
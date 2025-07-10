from flask import Flask, render_template, request
from predictor import predict_all, feature_names

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    user_id = None
    if request.method == "POST":
        user_id = request.form.get("user_id", None)
        try:
            input_data = {f: [float(request.form[f])] for f in feature_names}
            smoking_history = request.form["smoking_history"]
            gender = request.form["gender"]
            result = predict_all(input_data, smoking_history, gender)
        except Exception as e:
            result = {"Error": f"Invalid input: {e}"}
    return render_template("form.html", feature_names=feature_names, result=result, user_id=user_id)

if __name__ == "__main__":
    import webbrowser, threading
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True)
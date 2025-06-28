from flask import Flask, render_template_string, request
from predictor import predict_all, feature_names

app = Flask(__name__)

form_html = """
<!doctype html>
<title>Diabetes Classifier</title>
<h2>Enter Patient Data</h2>
<form method=post>
  {% for feature in feature_names %}
    <label>{{ feature }}: <input type="text" name="{{ feature }}"></label><br>
  {% endfor %}
  <label>smoking_history:
    <select name="smoking_history">
      <option value="never">never</option>
      <option value="No Info">No Info</option>
      <option value="current">current</option>
      <option value="former">former</option>
      <option value="not current">not current</option>
      <option value="ever">ever</option>
    </select>
  </label><br>
  <label>gender:
    <select name="gender">
      <option value="Male">Male</option>
      <option value="Female">Female</option>
      <option value="Other">Other</option>
    </select>
  </label><br>
  <input type=submit value=Predict>
</form>
{% if result %}
  <h3>Results:</h3>
  <ul>
    {% for k, v in result.items() %}
      <li><b>{{ k }}:</b> {{ v }}</li>
    {% endfor %}
  </ul>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        input_data = {f: [float(request.form[f])] for f in feature_names}
        smoking_history = request.form["smoking_history"]
        gender = request.form["gender"]
        result = predict_all(input_data, smoking_history, gender)
    return render_template_string(form_html, feature_names=feature_names, result=result)

if __name__ == "__main__":
    import webbrowser, threading
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True)
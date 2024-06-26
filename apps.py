from flask import Flask, request, render_template_string
from chatbot import ask_question
import pandas as pd

app = Flask(__name__)

# Read CSV file into a DataFrame for the initial table
table = pd.read_csv('sample jira extract.csv')
table = table.astype(str)  # Convert all data to strings to ensure consistency

# HTML template for the form with Bootstrap
form_template = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Jira Search Engine</title>
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body {
        padding-top: 50px;
      }
      .container {
        max-width: 600px;
      }
      .chatbox {
        background: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
      }
      .response {
        background: #e9ecef;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1 class="text-center">Jira AI</h1>
      <div class="chatbox">
        <form action="/" method="post">
          <div class="form-group">
            <label for="user_query">Enter your question:</label>
            <input type="text" class="form-control" id="user_query" name="user_query" required>
          </div>
          <button type="submit" class="btn btn-primary btn-block">Submit</button>
        </form>
        {% if response %}
        <div class="response mt-3">
          <h5>Response:</h5>
          <p>{{ response }}</p>
        </div>
        {% endif %}
      </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
  </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    response = None
    if request.method == 'POST':
        user_query = request.form.get('user_query', '')  # Get the user's query from the form
        if user_query.lower() == "exit":
            response = "Goodbye!"
        else:
            # Pass the DataFrame to ask_question
            response = ask_question(user_query, table)
    return render_template_string(form_template, response=response)

if __name__ == '__main__':
    print("Starting Flask server...")  # Debugging print statement
    app.run(host='127.0.0.1', port=5001, debug=True)  # Runs on localhost:5001 with debugging enabled

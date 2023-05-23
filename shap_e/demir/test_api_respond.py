from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/endpoint', methods=['POST'])
# def endpoint():
#     data = request.form  # Get the form data from the request
#     name = data.get('name')
#     email = data.get('email')

#     # Process the received data and prepare a response
#     response_data = {
#         'message': f'Hello, {name}! Your email is {email}.',
#         'status': 'success'
#     }

#     return jsonify(response_data)

# if __name__ == '__main__':
#     app.run()

app = Flask(__name__)

@app.route('/construct', methods=['POST'])
def process_sentence():
    sentence = request.form.get('sentence')
    # Do something with the received sentence
    # You can perform any necessary processing or logic here
    # For now, let's just print the sentence
    print(f"Received sentence: {sentence}")
    return 'Sentence received'

if __name__ == '__main__':
    app.run()

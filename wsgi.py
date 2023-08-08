import json
from flask import Flask, jsonify, request
from prediction import predict

application = Flask(__name__)


@application.route('/')
@application.route('/status')
def status():
    return jsonify({'status': 'ok'})


@application.route('/predictions', methods=['POST'])
def create_prediction():
    data = request.data or '{}'
    body = json.loads(data)
    return jsonify(predict(body))


@application.errorhandler(Exception)
def handle_exception(e):
    print(f'Exception occurred: {e}')
    response = {
        'success': False,
        'error': {
            'type': type(e).__name__,
            'message': str(e)
        }
    }
    return jsonify(response), 500

from flask import Flask, request, Response
import pickle
from predictor import ClientRanker
import pandas as pd
import os

model = pickle.load( open('../pkl/model_adaboost_trained.pkl', 'rb') )

app = Flask(__name__)

@app.route( '/rank_clients/rank', methods=['POST'] )
def rank_clients(model):
    clients_to_rank_json = request.get_json()
    
    if clients_to_rank_json:
        if isinstance( clients_to_rank_json, dict ): # unique example
            raw_data = pd.DataFrame( clients_to_rank_json, index=[0] )
            
        else: # multiple example
            raw_data = pd.DataFrame( clients_to_rank_json, columns=clients_to_rank_json[0].keys() )
            
        ranker = ClientRanker(model)

        df = ranker.predict(raw_data)
        
        return df
    
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    

    
if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5000)
    app.run( host='0.0.0.0', port=PORT )
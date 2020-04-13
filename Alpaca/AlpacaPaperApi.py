from alpaca_trade_api import REST
import json

class AlpacaPaperApi(REST):
    def __init__(self):
        with open('alpaca_creds.json') as json_creds:
            creds = json.load(json_creds)

        super().__init__(
            key_id=creds.key_id,
            secret_key=creds.secret_key,
            base_url=creds.base_url
        )
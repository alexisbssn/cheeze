from AlpacaPaperApi import AlpacaPaperApi
from ModelRunner import ModelRunner
import pandas as pd
import numpy as np
import time

class PortfolioManagementSystem:

    # TODO config
    def __init__(self) # , api, symbol, time_frame, system_id, system_label):
        # Connect to api
        # Connect to BrokenPipeError
        # Save fields to class

        # self.api = api
        self.api = AlpacaPaperApi()

        # self.symbol = symbol
        self.symbol = 'IBM'

        #self.time_frame = time_frame
        self.time_frame = 86400

        #self.system_id = system_id
        self.system_id = 1

        #self.system_label = system_label
        self.system_label = 'AI_PM'
        
        self.AI = AIModel()
        thread = threading.Thread(target=self.system_loop)
        thread.start()


    def place_buy_order(self):
        self.api.submit_order(
                        symbol='IBM',
                        qty=1,
                        side='buy',
                        type='market',
                        time_in_force='day',
                    )

    def place_sell_order(self):
        self.api.submit_order(
                        symbol='IBM',
                        qty=1,
                        side='sell',
                        type='market',
                        time_in_force='day',
                    )

    def system_loop(self):
        # Variables for weekly close
        this_weeks_close = 0
        last_weeks_close = 0
        delta = 0
        day_count = 0
        while(True):
            # Wait a day to request more data
            time.sleep(1440)
            # Request EoD data for IBM
            data_req = self.api.get_barset('IBM', timeframe='1D', limit=1).df
            # Construct dataframe to predict
            x = pd.DataFrame(
                data=[[
                    data_req['IBM']['close'][0]]], columns='Close'.split()
            )
            if(day_count == 7):
                day_count = 0
                last_weeks_close = this_weeks_close
                this_weeks_close = x['Close']
                delta = this_weeks_close - last_weeks_close

                # AI choosing to buy, sell, or hold
                if np.around(self.AI.network.predict([delta])) <= -.5:
                    self.place_sell_order()

                elif np.around(self.AI.network.predict([delta]) >= .5):
                    self.place_buy_order()

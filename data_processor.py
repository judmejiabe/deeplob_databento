#Generates a dataframe with the following columns: Timestamp, 
#best ask price, best bid price, volumes at 1, 2, ..., d levels of the LOB

__author__ = 'Juan Diego Mejia Becerra'
__email__ = 'judmejiabe@unal.edu.co'

import sys
import os
from pathlib import Path
import databento as db
import datetime
import copy

#Line of the day!!
sys.path.append('external/MeatPy')


from external.MeatPy.meatpy.event_handlers.lob_recorder import LOBRecorder
from external.MeatPy.meatpy.databento.databento_message_parser import DatabentoMessageParser
from external.MeatPy.meatpy.databento.databento_market_processor import \
    DatabentoMarketProcessor




class DeepLOBRecorder(LOBRecorder):

    max_price = 2 ** 32

    def __init__(self, L = 10):
        self.L = L
        self.last_timestamp = None
        super(DeepLOBRecorder, self).__init__(max_depth = L + 2)

    def record(self, lob, record_timestamp = None):

        new_record = {}
        current_lob = lob.copy(max_level = self.L + 2)

        if record_timestamp is not None:
            current_lob.timestamp = record_timestamp
            new_record['timestamp'] = record_timestamp

        else:
            new_record['timestamp'] = current_lob.timestamp

            if self.last_timestamp is not None:
                if self.last_timestamp == current_lob.timestamp:
                    self.records.pop()

            self.last_timestamp = current_lob.timestamp
        
        for l in range(self.L):

            try:
                new_record['ASKp' + str(l + 1)] = current_lob.ask_levels[l].price
            except IndexError:
                new_record['ASKp' + str(l + 1)] = self.max_price

            try:
                new_record['ASKs' + str(l + 1)] = current_lob.ask_levels[l].volume()
            except IndexError:
                new_record['ASKs' + str(l + 1)] = 0

            try:
                new_record['BIDp' + str(l + 1)] = current_lob.bid_levels[l].price
            except IndexError:
                new_record['BIDp' + str(l + 1)] = 0

            try:
                new_record['BIDs' + str(l + 1)] = current_lob.bid_levels[l].volume()
            except IndexError:
                new_record['BIDs' + str(l + 1)] = 0

        self.records.append(new_record)

    def write_csv(self, file):
        """Write records to a file in CSV format"""
        # Write header row
        record_keys = self.records[0].keys()
        header = ','.join(record_keys) + '\n'
        file.write(header)
        # Write content
        for x in self.records:
            row = (','.join([str(x[key]) for key in record_keys]) + '\n')
            file.write(row)

def record_data(mbo_data, date, stock):
    mbo_data_orig = copy.deepcopy(mbo_data)
    N = mbo_data_orig.shape[0]
    print('Messages to process:', N)

    #Create a parser object
    parser = DatabentoMessageParser()


    #Parse the stock data
    print('Parsing messages...')
    parser.parse(stock_data)
    print('Done parsing messages')

    processor = DatabentoMarketProcessor(stock, date)
    deeplob_recorder = DeepLOBRecorder()

    processor.handlers.append(deeplob_recorder)

    counter = 0
    stock_data_iter = iter(stock_data.iterrows())
    print('Processing messages...')
    for m in parser.messages:
        if counter % 10000 == 0:
            print('Processed', counter, 'messages out of', N, 'messages.')
        processor.process_message(m, stock_data_iter)
        counter += 1

    prefix = str(recorded_data_path) + '/' + date.strftime('%m%d%Y') + '_'
    with open(prefix + 'deeplob_data.csv', 'w') as output_file:
        deeplob_recorder.write_csv(output_file)




if __name__ == '__main__':
    data_path = Path('./data.nosync')

    raw_data_path = data_path / 'Raw_Data'
    raw_data_path.mkdir(parents=True, exist_ok=True) 

    # Recorded data directory
    recorded_data_path = data_path / 'Recorded_Data' 
    # Create the directory if it doesn't exist
    recorded_data_path.mkdir(parents=True, exist_ok=True) 

    for itch_data_path in os.listdir(str(raw_data_path)):
        # Extract the date from the file name
        date = itch_data_path.split('-')[2].split('.')[0]
        # Convert the date string to a datetime object
        
        date = datetime.datetime.strptime(date, '%Y%m%d').date()
        #print(type(date))
        stored_data = db.DBNStore.from_file(raw_data_path / itch_data_path)
        stock_data = stored_data.to_df()
        print('Processing data for', date)
        record_data(stock_data, date, 'MSFT')
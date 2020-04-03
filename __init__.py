from ModelBuilder import ModelBuilder
from ModelEvaluator import ModelEvaluator
from DataTransformer import csv_to_dataset
from ModelLoader import ModelLoader

dataset = csv_to_dataset('test_data/SHOP_daily.csv')
model_loader = ModelLoader()

#test_data = ModelBuilder().build_model(dataset, 150)
#model_loader.save_model(test_data.model, 'twobranch-2020-04-02')

test_data = ModelBuilder().split_test_data(dataset, 0.7)
test_data.model = model_loader.load_model('120-120.h5')

evaluator = ModelEvaluator()
evaluator.analyze(test_data)
evaluator.plot(test_data)
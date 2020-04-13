from ModelBuilder import ModelBuilder
from ModelEvaluator import ModelEvaluator
from DataTransformer import multi_csv_to_dataset
from ModelLoader import ModelLoader

dataset = multi_csv_to_dataset([
  'test_data/SHOP_daily.csv',
  # 'test_data/TD_daily.csv',
  # 'test_data/ENB_daily.csv',
  # 'test_data/BA_daily.csv',
  # 'test_data/TSLA_daily.csv'
  ])
model_loader = ModelLoader()

#test_data = ModelBuilder().build_model(dataset, 150)
#model_loader.save_model(test_data.model, 'multistock-2020-04-09')

test_data = ModelBuilder().split_test_data(dataset, 0.7)
test_data.model = model_loader.load_model('multistock-2020-04-09.h5')

evaluator = ModelEvaluator()
evaluator.analyze(test_data)
evaluator.plot(test_data)
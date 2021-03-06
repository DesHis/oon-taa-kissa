from imageai.Prediction.Custom import ModelTraining
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory(r"C:\Users\Altti\Desktop\tekoäly")
model_trainer.trainModel(num_objects=2, num_experiments=200, enhance_data=True, batch_size=16, show_network_summary=True)
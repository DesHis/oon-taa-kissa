from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("C:\\Users\\Altti\\Desktop\\model_ex-085_acc-0.966261.h5")
prediction.setJsonPath("C:\\Users\\Altti\\Desktop\\model_class.json")
prediction.loadModel(num_objects=2)

predictions, probabilities = prediction.predictImage("C:\\Users\\Altti\\Desktop\\testi12.png", result_count=2)

p1, p2 = zip(predictions, probabilities)

varmuus=str(p1[1])[:4]

if p1[0]=="kissa":
    print("oon tää kissa :D ("+varmuus+"% varmuus)")
else:
    print("en oo tää kissa :( (" + varmuus + "% varmuus)")

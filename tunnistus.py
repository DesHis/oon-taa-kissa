from imageai.Prediction.Custom import CustomImagePrediction
import os

print('Kuvan täytyy olla samassa kansiossa kuin tämän python tiedoston.')
print('Syötä kuvan nimi (esim kuva.jpg): ')

execution_path = os.getcwd()
kuva = input()
dir = os.path.dirname(os.path.realpath(__file__))
dir.replace("\\","\\\\")

print("")
print("Analysoidaan kuvaa...")
print("")

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(dir+"\\model_ex-085_acc-0.966261.h5")
prediction.setJsonPath(dir+"\\model_class.json")
prediction.loadModel(num_objects=2)


predictions, probabilities = prediction.predictImage(dir+"\\"+kuva, result_count=2)

p1, p2 = zip(predictions, probabilities)

varmuus=str(p1[1])[:4]

print("")

if p1[0]=="kissa":
    print("oon tää kissa :D ("+varmuus+"% varmuus)")
else:
    print("en oo tää kissa :( (" + varmuus + "% varmuus)")

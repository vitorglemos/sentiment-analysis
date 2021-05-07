from tensorflow.keras.models import load_model
import operator

models_h5 = ["mv1.h5", "mv2.h5", "mv3.h5", "mv4.h5"]

models = {}

x_test = np.load('./mod_xtest.npy')
y_test = np.load('./mod_ytest.npy')

for model in models_h5:
  model = load_model(model)
  scores = model.evaluate(np.array(x_test), np.array(y_test), batch_size=64)
  print(f"Log-loss: {scores[0]}  Acc: {scores[1]} 
        

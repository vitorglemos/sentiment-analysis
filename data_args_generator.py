from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
      rotation_range=30,       # valor de alcance para randomicamente rotacionar a imagem (rotacionar no maximo 30 graus)
      shear_range=0.1,         # aleatoriamente distorce a imagem (por cisalhamento)
      zoom_range=0.3,          # zoom in image
      width_shift_range=0.1,   # aleatoriamente alterna as imagens horizontalmente (fração da largura total)
      height_shift_range=0.1,  # aleatoriamente alterna as imagens verticalmente (largura total)
      horizontal_flip=True,    # aleatoriamente vira as imagens na horizontal
      vertical_flip=False,     # aleatoriamente vira as imagens na vertical 
      #rescale=1./255,         # normalização, descomentar se já não estiver feito
      fill_mode='nearest')     # define como vai preencher espaços fora do limite da imagem ('nearest' é o valor padrão)


print(len(datagen.flow(x_train, y_train)))


model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

path_model = "mv1_dataarg.h5" # model
path_model_json = "mv1_dataarg.json" # architecture

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(arquivo_modelo, monitor='val_loss', verbose=1, save_best_only=True)

model_json = model.to_json()
with open(path_model_json, "w") as json_file:
    json_file.write(model_json)
    
# use fit_generator()
batch_size = 64
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
          epochs=100,
          verbose=1,
          validation_data=(x_val, y_val),
          validation_steps = len(x_val) // batch_size,
          steps_per_epoch=len(x_train) // batch_size,
          callbacks=[lr_reducer, early_stopper, checkpointer])

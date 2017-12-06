import datetime
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Funcion para preparar el dataset
def prepare_dataset(data, headers_feature):
    # La data de entrenamiento y de pruebas es obtenida a razon de 70% / 30%
    train, test = data.randomSplit(
        [0.7, 0.3], seed=12345
    )
    # train.show()
    header_output = "features"

    # Generar los vectores de la data de entrenamiento y pruebas
    # La caracteristica a predecir es 'genre'
    assembler = VectorAssembler(
        inputCols=headers_feature,
        outputCol=header_output)
    train_data = assembler.transform(train).select("features", "label")
    test_data = assembler.transform(test).select("features", "label")

    return train_data, test_data

def run(data, headers_feature):
    print('-*-*-*- Iniciando la red neuronal -*-*-*-')
    start_time = datetime.datetime.now()
    print('Tiempo inicial', start_time)

    # Obtener data de entrenamiento y pruebas
    train_data, test_data = prepare_dataset(data, headers_feature)
    train_data.show()

    # Especificar las capas para la red neuronal:
    # input de 5000 (features),
    # 5 capas intermedias de 50 neuronas
    # y output de 101 (clases)
    layers = [5000, 50, 50, 50, 50, 50, 101]

    # Configurar la el clasificador perceptron multicapa
    mlpc = MultilayerPerceptronClassifier(
        maxIter=100, layers=layers, blockSize=128, seed=1234
    )

    # Obtener el modelo de clasificacion
    mlpc_model = mlpc.fit(train_data)

    # print("Coeficientes: " + str(lr_model.coefficients))
    # print("Intercepto: " + str(lr_model.intercept))

    data_to_validate = mlpc_model.transform(test_data)

    # Validar la precision de la prediccion de la data de prueba
    prediction = data_to_validate.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Precision de la prueba = " + str(evaluator.evaluate(prediction)))

    end_time = datetime.datetime.now()
    print('Tiempo final', end_time)
    print('Tiempo transcurrido para red neuronal', end_time - start_time)

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

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
    train_data = assembler.transform(train).select("features", "genre")
    test_data = assembler.transform(test).select("features", "genre")

    return train_data,test_data

def run(data, headers_feature):
    # Obtener data de entrenamiento y pruebas
    train_data, test_data = prepare_dataset(data, headers_feature)

    # Configurar la regresion logistica multinomial
    lr = LogisticRegression(
        maxIter=200, regParam=0.3, elasticNetParam=0.8,
        labelCol='genre', family='multinomial'
    )
    # LogisticRegression: All labels are the same value and fitIntercept=true, so the coefficients will be zeros. Training is not needed.

    # Obtener el modelo de clasificacion
    lr_model = lr.fit(train_data)

    # print("Coeficientes: " + str(lr_model.coefficients))
    # print("Intercepto: " + str(lr_model.intercept))

    data_to_validate = lr_model.transform(test_data)

    evaluator1 = BinaryClassificationEvaluator(
        labelCol='genre', metricName='areaUnderROC',
        rawPredictionCol='rawPrediction'
    )
    ROC = evaluator1.evaluate(data_to_validate)
    print("{}:{}".format("areaUnderROC",ROC))

    evaluator2 = BinaryClassificationEvaluator(
        labelCol='genre', metricName='areaUnderPR',
        rawPredictionCol='rawPrediction'
    )
    PR = evaluator2.evaluate(data_to_validate)
    print("{}:{}".format("areaUnderPR",PR))

    return ROC, PR

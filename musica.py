import sqlite3
import datetime
import regression
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
from math import log

# Funcion que inicializa la configuracion de spark y retorna el contexto
def init_spark():
    conf = SparkConf().setAppName("Music").setMaster("local")
    return SparkContext(conf=conf)

# Funcion que inicializa conexion a sqlite
def init_sqlite():
    con = sqlite3.connect("./ABD_FINAL/mxm_dataset.db")
    return con.cursor()

def load_genres(c):
    distinct_genres = c.execute('SELECT DISTINCT genre FROM songs').fetchall()
    genres = []
    for i, genre in enumerate(distinct_genres):
        genres.append((i, genre[0]))
    return genres

def get_genre(genres, genre):
    for g in genres:
        if g[1] == genre:
            return g[0]
    return -1

def tfidf(c, tf, n, word):
    df = len(c.execute(
        'SELECT * FROM lyrics WHERE word=:word',
        {'word': word}
    ).fetchall())
    return tf * log(n / df)

# Funcion que carga la data en un dataframe
def load_dataset(c, spark, genres):
    headers = ['mxm_tid']
    for word in c.execute('SELECT word FROM words'):
        headers.append(word[0])
    headers.append('genre')

    n = len(c.execute('SELECT * FROM songs').fetchall())

    data = []
    for song in c.execute('SELECT mxm_tid, genre FROM songs'):
        row = []
        row.append(song[0])
        for i in range(1, len(headers) - 1):
            count = c.execute(
                'SELECT count FROM lyrics WHERE mxm_tid=:song AND word=:word',
                {'song': song[0], 'word': headers[i]}
            ).fetchone()
            if not count:
                row.append(0)
            else:
                row.append(tfidf(c, count[0], n, headers[i]))
        row.append(get_genre(genres, song[1]))
        data.append(row)

    headers_feature = []
    for word in headers[1:len(headers)-1]:
        headers_feature.append(word.replace('`', ''))

    return spark.createDataFrame(data, headers), headers_feature

def main ():
    start_time = datetime.datetime.now()
    print('Tiempo inicial', start_time)

    #Iniciar Spark
    sc = init_spark()
    spark = SparkSession(sc)

    # Conexion a sqlite
    c = init_sqlite()

    # Generar una lista con todos los generos y asignarle un numero a cada uno
    genres = load_genres(c)

    # Cargar la data a un dataframe
    data, headers_feature = load_dataset(c, spark, genres)

    # Correr la clasificacion
    regression.run(data, headers_feature)

    end_time = datetime.datetime.now()
    print('Tiempo final', end_time)
    print('Tiempo total transcurrido', end_time - start_time)

if __name__ == '__main__' :
    main()

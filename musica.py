import sqlite3
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

def tfidf(tf, n, df):
    return tf * log(n / df)

# Funcion que carga la data en un dataframe
def load_dataset(c, spark, genres):
    # sacar data  y guardarlo en arreglos / el fetchall es para jalar toda la lista
    # songs = c.execute("SELECT mxm_tid, genre FROM songs").fetchall()
    # words_table = c.execute("SELECT word FROM words").fetchall()
    # lyrics = c.execute("SELECT track_id, mxm_tid, word, count, is_test FROM lyrics").fetchall()

    # Guardar las cabeceras correspondientes a los campos
    headers = ['mxm_tid']
    for word in c.execute('SELECT word FROM words'):
        headers.append(word[0])
    headers.append('genre')

    data = []
    for song in c.execute('SELECT mxm_tid, genre FROM songs'):
        row = []
        row.append(song[0])
        for i in range(1, len(headers) - 1):
            count = c.execute(
                'SELECT count from lyrics where mxm_tid=:song AND word=:word',
                {'song': song[0], 'word': headers[i]}
            ).fetchone()
            if not count:
                row.append(0)
            else:
                row.append(count[0])
        row.append(get_genre(genres, song[1]))
        data.append(row)

    return spark.createDataFrame(data, headers)

def main ():
    #Iniciar Spark
    sc = init_spark()
    spark = SparkSession(sc)

    # Conexion a sqlite
    c = init_sqlite()

    # Generar una lista con todos los generos y asignarle un numero a cada uno
    genres = load_genres(c)

    # Cargar la data a un dataframe
    data = load_dataset(c, spark, genres)

if __name__ == '__main__' :
    main()

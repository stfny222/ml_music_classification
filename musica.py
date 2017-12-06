import sqlite3
import datetime
import regression
import neural
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from math import log

# Funcion que inicializa la configuracion de spark y retorna el contexto
def init_spark():
    conf = SparkConf().setAppName("Music").setMaster("local")
    return SparkContext(conf=conf)

# Funcion que inicializa conexion a sqlite
def init_sqlite():
    con = sqlite3.connect("./ABD_FINAL/mxm_dataset.db")
    return con.cursor()

# Funcion que almacena todos los generos y les asigna un identificador numerico
def load_genres(c):
    # Consultar los generos distintos en la tabla songs
    distinct_genres = c.execute('SELECT DISTINCT genre FROM songs').fetchall()
    # Inicializar un arreglo para almacenar los generos
    genres = []
    for i, genre in enumerate(distinct_genres):
        # Almacenar una tupla (identificador, genero)
        genres.append((i, genre[0]))
    return genres

# Funcion que devuelve el identificador de un genero
def get_genre(genres, genre):
    # recorrer la lista de tuplas
    for g in genres:
        # si el valor de la tupla es el genero buscado, se devuelve el identificador
        if g[1] == genre:
            return g[0]
    return -1

# Funcion para calcular el tf_idf, la frecuencia de una palabra en una cancion
def tfidf(c, tf, n, word):
    # Consultar la cantidad de canciones que contienen la palabra
    df = len(c.execute(
        'SELECT * FROM lyrics WHERE word=:word',
        {'word': word}
    ).fetchall())
    # Retorna el producto del tf por el idf
    return tf * log(n / df)

# Funcion que carga la data en un dataframe
def load_dataset(c, spark, genres):
    print('-*-*-*- Generando el dataframe -*-*-*-')
    # Inicializar un arreglo para las cabeceras. La primera corresponde al codigo de la cancion
    headers = ['mxm_tid']
    # Inicializar un arreglo vacio para las cabeceras de las caracteristicas
    headers_feature = []
    # Recorrer los resultados del query para todas las palabras de la tabla words
    for word in c.execute('SELECT word FROM words'):
        # Agregar cada palabra a la lista de caracteristicas
        # Se necesito reemplazar el caracter '`'
        # pues no era valido dentro del nombre de la caracteristica
        headers_feature.append(word[0].replace('`', ''))
        # Agregar cada palabra a la lista total de cabeceras
        headers.append(word[0])
    # Agregar como ultima cabecera el genero de la cancion
    headers.append('genre')

    print('Cantidad de caracteristicas:', len(headers_feature))

    # Consultar el total de canciones de la tabla songs
    songs = c.execute('SELECT mxm_tid, genre FROM songs').fetchall()
    # Obtener el numero total de canciones de la tabla songs
    n = len(songs)
    print('Numero total de canciones:', n)

    # Generar un arreglo vacio para almacenar toda la data
    data = []
    # Recorrer todas las canciones de la tabla songs
    # (Se realizo una prueba con solo 10 registros)
    # for song in songs[:10]:
    for song in songs:
        # Generar un arreglo vacio para almacenar la data referente a una sola cancion
        row = []
        # El primer valor es el codigo de la cancion (mxm_tid)
        row.append(song[0])
        # Por cada cancion, recorrer la lista con todas las palabras
        for i in range(1, len(headers) - 1):
            # Ejecutar una consulta en la tabla lyrics donde se encuentre
            # la cancion con la palabra actual
            count = c.execute(
                'SELECT count FROM lyrics WHERE mxm_tid=:song AND word=:word',
                {'song': song[0], 'word': headers[i]}
            ).fetchone()
            # Si la consulta no devuelve nada, se asigna el valor 0
            if not count:
                row.append(float(0))
            # De lo contrario, se asigna el valor tf_idf para esta caracteristica
            else:
                row.append(tfidf(c, count[0], n, headers[i]))
        # El ultimo valor del registro es el genero de la cancion, representado por un numero
        row.append(get_genre(genres, song[1]))
        # Cada registro se va agregando al arreglo que almacena la data
        data.append(row)

    print('Cantidad de datos guardados:', len(data))
    # Esta funcion retorna el dataframe creado y la lista de caracteristicas (palabras)
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

    # Correr la clasificacion por regresion logistica
    regression.run(data, headers_feature)

    # Correr la clasificacion por redes neuronales
    neural.run(data, headers_feature)

    end_time = datetime.datetime.now()
    print('Tiempo final', end_time)
    print('Tiempo total transcurrido', end_time - start_time)

if __name__ == '__main__' :
    main()

from flask import Flask, request, render_template, redirect
import recomendador 

app = Flask(__name__)

@app.get('/')
def get_index():
    return render_template('login.html')

@app.post('/')
def post_index():
    id_lector = request.form.get('id_lector')
    algoritmo = request.form.get('algoritmo', recomendador.ALGORITMO_POR_DEFECTO)
    if not id_lector:
        return render_template('login.html')
    # TODO: Validar que el id_lector exista en la base de datos
    return redirect(f'/recomendar/{id_lector}/{algoritmo}')


@app.get('/reset/<id_lector>/<algoritmo>')
def get_reset(id_lector, algoritmo):
    recomendador.resetear_recientes(id_lector)
    return redirect(f'/recomendar/{id_lector}/{algoritmo}')


@app.get('/recomendar/<id_lector>/<algoritmo>')
def get_recomendar(id_lector, algoritmo):
    libros_recomendados = recomendador.recomendar(id_lector, algoritmo)
    cant_vistos = recomendador.items_vistos(id_lector)[0]['items_vistos']
    cant_valorados = recomendador.item_valorados(id_lector)[0]['item_valorados']
    return render_template('recomendaciones.html', cant_vistos=cant_vistos, cant_valorados=cant_valorados, libros_recomendados=libros_recomendados, id_lector=id_lector, algoritmo=algoritmo)


@app.get('/predecir/<algoritmo>')
def get_predecir(algoritmo):
    # genera data/prediction_<algoritmo>.csv a partir de data/prediction_base.csv
    if algoritmo not in recomendador.ALGORITMOS:
        return {'error': f'Algoritmo desconocido: {algoritmo}', 'algoritmos': list(recomendador.ALGORITMOS)}, 400
    archivo, cant_lectores = recomendador.generar_predicciones(algoritmo)
    return {'algoritmo': algoritmo, 'archivo': archivo, 'lectores': cant_lectores}


@app.post('/recomendar/<id_lector>/<algoritmo>')
def post_recomendar(id_lector, algoritmo):
    # guardar los ítems puntuados, y volver a recomendar
    # los vistos los guardo en el get_recomendar, porque ahí es donde se muestran
    for id_libro in request.form.keys():
        rating = int(request.form.get(id_libro, 0))
        print(f'El lector {id_lector} puntuó el libro {id_libro} con {rating}')
        if rating > 0: # Lo puntuó, entonces lo guardamos en la tabla de interacciones
            recomendador.insertar_valorado(id_lector, id_libro, rating)
        
    return redirect(f'/recomendar/{id_lector}/{algoritmo}')
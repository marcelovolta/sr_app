from flask import Flask, request, render_template, redirect
import recomendador 

app = Flask(__name__)

@app.get('/')
def get_index():
    return render_template('login.html')

@app.post('/')
def post_index():
    id_lector = request.form.get('id_lector')
    if not id_lector:
        return render_template('login.html')
    # TODO: Validar que el id_lector exista en la base de datos
    return redirect(f'/recomendar/{id_lector}')


@app.get('/reset/<id_lector>')
def get_reset(id_lector):
    recomendador.resetear_recientes(id_lector)
    return redirect(f'/recomendar/{id_lector}')


@app.get('/recomendar/<id_lector>')
def get_recomendar(id_lector):
    libros_recomendados = recomendador.recomendar(id_lector)
    cant_vistos = recomendador.items_vistos(id_lector)[0]['items_vistos']
    cant_valorados = recomendador.item_valorados(id_lector)[0]['item_valorados']
    return render_template('recomendaciones.html', cant_vistos=cant_vistos, cant_valorados=cant_valorados, libros_recomendados=libros_recomendados, id_lector=id_lector)


@app.post('/recomendar/<id_lector>')
def post_recomendar(id_lector):
    # guardar los ítems puntuados, y volver a recomendar
    # los vistos los guardo en el get_recomendar, porque ahí es donde se muestran
    for id_libro in request.form.keys():
        rating = int(request.form.get(id_libro, 0))
        print(f'El lector {id_lector} puntuó el libro {id_libro} con {rating}')
        if rating > 0: # Lo puntuó, entonces lo guardamos en la tabla de interacciones
            recomendador.insertar_valorado(id_lector, id_libro, rating)
        
    return redirect(f'/recomendar/{id_lector}')

from itertools import product

import numpy as np

LUGARES = {
    "Barcelona": ["Montserrat", "la Costa Brava", "el Montseny", "Sitges", "el Penedès", "Collserola"],
    "Madrid": ["la Sierra de Guadarrama", "Toledo", "Segovia", "Aranjuez", "Chinchón", "el Valle del Lozoya"],
    "Seville": ["Carmona", "Itálica", "la Sierra Norte", "el Guadalquivir", "Osuna", "las Marismas"],
    "Valencia": ["la Albufera", "Sagunto", "la Sierra Calderona", "Xàtiva", "Requena", "la huerta valenciana"],
    "Malaga": ["el Caminito del Rey", "Ronda", "Nerja", "Frigiliana", "la Axarquía", "los Montes de Málaga"],
    "Palma de Mallorca": ["la Serra de Tramuntana", "Sóller", "Valldemossa", "Es Trenc", "Formentor", "la bahía de Palma"],
    "Tenerife": ["el Teide", "Masca", "Anaga", "Los Gigantes", "Garachico", "el Valle de La Orotava"],
    "Gran Canaria": ["el Roque Nublo", "las dunas de Maspalomas", "Agaete", "Teror", "el barranco de Guayadeque", "Tejeda"],
}

HITOS = {
    "Barcelona": ["la Sagrada Família", "el Park Güell", "el Barrio Gótico", "la Casa Batlló", "el Palau de la Música"],
    "Madrid": ["el Museo del Prado", "el Palacio Real", "el Madrid de los Austrias", "el Retiro", "el Reina Sofía"],
    "Seville": ["el Real Alcázar", "la Catedral y la Giralda", "la Plaza de España", "Triana", "el barrio de Santa Cruz"],
    "Valencia": ["la Ciudad de las Artes", "la Lonja de la Seda", "el barrio del Carmen", "el Mercado Central", "la Catedral"],
    "Malaga": ["la Alcazaba", "el Museo Picasso", "el Teatro Romano", "el Castillo de Gibralfaro", "el centro histórico"],
    "Palma de Mallorca": ["la Catedral de Palma", "el Castillo de Bellver", "el casco antiguo", "la Lonja", "los patios de Palma"],
    "Tenerife": ["La Laguna", "el Auditorio de Tenerife", "Puerto de la Cruz", "La Orotava", "Santa Cruz"],
    "Gran Canaria": ["Vegueta", "la Casa de Colón", "la Catedral de Santa Ana", "el Puerto de Las Palmas", "Triana"],
}

BARRIOS = {
    "Barcelona": ["el Born", "Gràcia", "la Barceloneta"],
    "Madrid": ["La Latina", "Malasaña", "Lavapiés"],
    "Seville": ["Triana", "la Alameda", "el Arenal"],
    "Valencia": ["Ruzafa", "el Carmen", "el Cabanyal"],
    "Malaga": ["el Soho", "el centro histórico", "Pedregalejo"],
    "Palma de Mallorca": ["Santa Catalina", "el casco antiguo", "el Portixol"],
    "Tenerife": ["La Laguna", "el barrio de El Toscal", "Puerto de la Cruz"],
    "Gran Canaria": ["Vegueta", "Triana", "Las Canteras"],
}

GASTRO = {
    "Barcelona": ["vinos del Penedès", "cocina catalana", "vermut y conservas"],
    "Madrid": ["cocido madrileño", "tapas castizas", "vinos de la sierra"],
    "Seville": ["jamón ibérico", "cocina andaluza", "vinos de Jerez"],
    "Valencia": ["paella valenciana", "horchata y fartons", "arroces tradicionales"],
    "Malaga": ["espetos y pescaíto", "vinos dulces de Málaga", "aceite de oliva de la Axarquía"],
    "Palma de Mallorca": ["ensaimadas", "vinos de Binissalem", "sobrasada mallorquina"],
    "Tenerife": ["mojos y papas arrugadas", "vinos volcánicos", "quesos canarios"],
    "Gran Canaria": ["mojo canario", "quesos de flor", "ron y café de Agaete"],
}

VEHICULOS = ["en minivan", "en sedán premium", "en vehículo eléctrico", "en autobús lanzadera",
             "con silla infantil incluida", "en monovolumen para grupos", "en taxi oficial"]

BASES = {
    "Excursions & Day Trips": [
        ("Excursión a {l}", "Escapada de un día a {l} con transporte de ida y vuelta."),
        ("Ruta guiada por {l}", "Recorrido guiado por {l} con paradas panorámicas."),
        ("Senderismo por {l}", "Caminata de dificultad moderada por los senderos de {l}."),
        ("Paseo en barco por {l}", "Navegación tranquila con vistas a {l}."),
        ("Ruta en bicicleta por {l}", "Pedalea por {l} con bicicleta y casco incluidos."),
        ("Excursión al atardecer a {l}", "Vive la puesta de sol en {l} en grupo reducido."),
        ("Safari fotográfico por {l}", "Captura los mejores rincones de {l} con un guía fotógrafo."),
        ("Escapada rural a {l}", "Jornada completa en {l} con tiempo libre para explorar."),
    ],
    "Local Experiences": [
        ("Taller de {g}", "Aprende a preparar {g} con un cocinero local."),
        ("Cata de {g}", "Degustación comentada de {g} en un local con encanto."),
        ("Ruta de tapas por {b}", "Recorrido gastronómico por los bares clásicos de {b}."),
        ("Clase de cocina: {g}", "Sesión práctica para dominar {g}, con receta incluida."),
        ("Visita al mercado y degustación de {g}", "Paseo entre puestos históricos con parada para probar {g}."),
        ("Encuentro con artesanos de {b}", "Conoce talleres locales de {b} y su oficio tradicional."),
        ("Velada tradicional en {b}", "Noche de música y cultura local en {b}."),
        ("Experiencia gastronómica: {g}", "Menú degustación en torno a {g} con maridaje."),
    ],
    "Attractions & Guided Tours": [
        ("Visita guiada a {h}", "Descubre {h} con un guía oficial y grupo reducido."),
        ("Free tour por {h}", "Paseo introductorio por {h} con anécdotas e historia."),
        ("Tour nocturno por {h}", "Recorre {h} iluminado, lejos de las aglomeraciones."),
        ("Tour privado por {h}", "Visita exclusiva a {h} adaptada a tu ritmo."),
        ("Visita con acceso preferente a {h}", "Entra sin esperas a {h} con entrada incluida."),
        ("Recorrido histórico por {h}", "Viaja al pasado de {h} de la mano de un historiador."),
        ("Tour de leyendas y misterios por {h}", "Historias sorprendentes en torno a {h} al caer la tarde."),
    ],
    "Tickets and Events": [
        ("Entrada a {h}", "Acceso general a {h} con horario a tu elección."),
        ("Entrada sin colas a {h}", "Acceso prioritario a {h}, sin esperas."),
        ("Concierto en {b}", "Música en directo en una sala emblemática de {b}."),
        ("Espectáculo tradicional en {b}", "Función de música y danza tradicional en {b}."),
        ("Festival gastronómico: {g}", "Jornada festiva en torno a {g} con degustaciones."),
        ("Entrada combinada: {h}", "Ticket combinado que incluye {h} y museos asociados."),
        ("Noche de música en vivo en {b}", "Sesión íntima de artistas locales en {b}."),
    ],
    "Transfers": [
        ("Traslado privado aeropuerto-hotel {v}", "Conductor esperándote a la llegada, trayecto directo {v}."),
        ("Traslado compartido aeropuerto-ciudad {v}", "Opción económica con paradas limitadas, {v}."),
        ("Traslado al puerto de cruceros {v}", "Llega a tu crucero sin estrés, {v}."),
        ("Chófer por horas {v}", "Vehículo con conductor a tu disposición, {v}."),
        ("Traslado nocturno {v}", "Servicio disponible en horario nocturno, {v}."),
        ("Traslado de ida y vuelta al aeropuerto {v}", "Ambos trayectos cubiertos, {v}."),
    ],
}

SUFIJOS = ["", " en grupo reducido", " con guía local", " con degustación incluida",
           " con recogida en el hotel", " para toda la familia", " al amanecer", " edición premium"]
SUFIJOS_TRF = ["", " 24 horas", " con seguimiento de vuelo", " puerta a puerta",
               " con espera incluida", " con asistencia a la llegada", " con cancelación gratuita"]


def generar_nombres(cat_df, seed: int = 42):
    """Devuelve (nombres, descripciones) alineados con el DataFrame del catálogo."""
    rng = np.random.default_rng(seed)
    nombres = [None] * len(cat_df)
    descs = [None] * len(cat_df)

    for (categ, dest), grupo in cat_df.groupby(["Category", "Destination"]).groups.items():
        slots = {"l": LUGARES[dest], "h": HITOS[dest], "b": BARRIOS[dest],
                 "g": GASTRO[dest], "v": VEHICULOS}
        sufijos = SUFIJOS_TRF if categ == "Transfers" else SUFIJOS
        combos = []
        for (plantilla, desc), suf in product(BASES[categ], sufijos):
            # detectar el slot usado por la plantilla
            slot = next(k for k in ["l", "h", "b", "g", "v"] if "{" + k + "}" in plantilla)
            for valor in slots[slot]:
                combos.append((plantilla.format(**{slot: valor}) + suf,
                               desc.format(**{slot: valor}) + (" " + dest + "." if categ == "Transfers" else "")))
        orden = rng.permutation(len(combos))
        for pos, idx in enumerate(grupo):
            n, d = combos[orden[pos % len(combos)]]
            nombres[idx] = n
            descs[idx] = d
    return nombres, descs

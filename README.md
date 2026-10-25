# Clasificador de cielos

Aplicación de escritorio en Python que segmenta el cielo de fotografías, extrae características de color y textura, agrupa las imágenes con k-means y las clasifica en categorías meteorológicas (despejado, semi-nublado y nublado). También detecta imágenes fuera de dominio con la distancia de Mahalanobis.

Proyecto final de Visión Artificial, ESCOM-IPN.

## Características

- **Segmentación del cielo** en espacio de color HSV, con máscara, morfología y relleno de huecos.
- **Extracción de características:** histogramas por canal, medias y desviaciones estándar, porcentajes de azul, gris y blanco, y textura con patrones binarios locales (LBP).
- **Agrupamiento con k-means implementado desde cero**, con inicialización k-means++ y múltiples inicios.
- **Asignación de clases de cielo** con funciones sigmoides suaves sobre los centros de cada grupo.
- **Clasificación y detección de outliers** con distancia de Mahalanobis, con umbral configurable.
- **Métricas de evaluación implementadas a mano:** silhouette, Davies-Bouldin y Calinski-Harabasz, además de F1, precisión y recall por clase.
- **Análisis de características:** correlación, importancia por F-score y varianza explicada con PCA.
- **Interfaz gráfica en PyQt5** y exportación de reportes en PDF.

## Requisitos

- Python 3.9 o superior
- Dependencias listadas en `requirements.txt` (scipy es opcional)

## Instalación

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python clasificador_cielos.py
```

1. Selecciona la carpeta con las imágenes (jpg, jpeg, png, bmp o webp).
2. Elige el número de clusters (k) y los nombres de las clases.
3. Ejecuta el análisis y revisa las gráficas de evaluación.
4. Exporta el reporte en PDF.

## Datos

La carpeta `data/features_cielos.csv` contiene la tabla de características extraídas de 899 imágenes, con el cluster y la clase asignada a cada una. Las imágenes originales no están incluidas en el repositorio.

## Resultados

[Completar con las métricas finales del reporte: silhouette, Davies-Bouldin, Calinski-Harabasz y F1 por clase.]

## Limitaciones

- La clasificación depende de la calidad de la segmentación del cielo. Las fotos con edificios o árboles grandes pueden afectar el resultado.
- El número de clases y sus nombres los define el usuario; el sistema no los aprende de forma supervisada.

## Autor

Sergio Alejandro Vera Martínez, ESCOM-IPN.

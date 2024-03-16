# ConvRFF - Implementación de modelos con Random Fourier Features in Convolutional Form

Este repositorio contiene la implementación de modelos de segmentación utilizando la técnica Random Fourier Features in Convolutional Form (ConvRFF), que incluye arquitecturas como Unet, ResUnet y FCN.

## Descripción

La técnica ConvRFF es una forma eficiente de aproximar las operaciones de convolución mediante la proyección de los filtros convolucionales en el espacio de características de Fourier aleatorias. Esto permite acelerar el cálculo de convoluciones en redes neuronales convolucionales (CNN) y reduce significativamente el costo computacional en comparación con el uso de convoluciones tradicionales.

En este repositorio, encontrarás implementaciones de modelos de segmentación utilizando la técnica ConvRFF, incluyendo:

- Unet con ConvRFF
- ResUnet con ConvRFF
- FCN (Fully Convolutional Network) con ConvRFF

## Contenido

El contenido de este repositorio se organiza de la siguiente manera:

- **convrff_models/**: Carpeta que contiene las implementaciones de modelos de segmentación utilizando ConvRFF.
- **utils/**: Carpeta que incluye las funciones y clases necesarias para la utilización de la técnica ConvRFF.
- **requirements.txt**: Archivo que lista las dependencias del proyecto.

## Uso

Para utilizar los modelos de segmentación implementados en este repositorio, sigue estos pasos:

1. Clona el repositorio a tu máquina local:

```bash
git clone https://github.com/aguirrejuan/ConvRFF.git

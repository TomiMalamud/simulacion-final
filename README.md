# Examen Final de Simulación -- Ejercicio 231

Este repositorio contiene una app web desarrollada con Flask, con resolución de EDO de orden superior con Runge-Kutta de 4to orden.

## Requisitos previos

Asegurarse de tener instalado lo siguiente en el sistema:

- Python 3.7+
- Visual Studio Code
- Extensión de Python para VS Code

## Instalación

1. Clonar este repositorio:
   - Abrir VS Code
   - Presionar `Ctrl+Shift+P` para abrir la paleta de comandos
   - Escribir "Git: Clone" y seleccionar la opción
   - Pegar la URL del repositorio: `https://github.com/TomiMalamud/simulacion-final`
   - Seleccionar una carpeta local para clonar el repositorio (normalmente Descargas)

2. Abrir la carpeta del proyecto en VS Code:
   - Ir a Archivo > Abrir Carpeta
   - Seleccionar la carpeta del repositorio clonado

3. Crear y activar un entorno virtual (opcional pero recomendado):
   - Abrir la paleta de comandos (Ctrl+Shift+P)
   - Escribir "Python: Create Environment" y seleccionar "Venv"
   - Elegir el intérprete de Python

4. Instalar las dependencias:
   - Abrir la paleta de comandos (Ctrl+Shift+P)
   - Escribir "Python: Run Python File in Terminal"
   - En el terminal integrado que se abre, escribir:
     ```
     pip install -r requirements.txt
     ```

## Ejecución

Para ejecutar la aplicación localmente:

1. Abrir el archivo `app.py` en VS Code

2. Presionar F5 o usar el menú Run > Start Debugging

3. VS Code debería abrir automáticamente un navegador. Si no, abrir manualmente el navegador y visitar `http://127.0.0.1:5000`

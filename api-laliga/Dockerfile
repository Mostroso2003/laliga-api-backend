# Usar una imagen base de Python oficial
FROM python:3.9-slim

# Establecer el directorio de trabajo en la imagen
WORKDIR /code

# Copiar el archivo de dependencias
COPY ./requirements.txt /code/requirements.txt

# Instalar las dependencias
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copiar los artefactos y el código de la aplicación
COPY ./artifacts /code/artifacts
COPY ./app /code/app

# Exponer el puerto en el que correrá la API
EXPOSE 8000

# Comando para correr la aplicación usando Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
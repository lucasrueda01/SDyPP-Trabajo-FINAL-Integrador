# TP Integrador Sistemas Distribuidos y Programación Paralela - UNLu

**Alumnos:** 
- Rueda Lucas Laureano, Leg. 169202
- Cosentino Matias, Leg. 166741
------------------------------------------------------------------------

## Descripción

Este trabajo implementa una arquitectura blockchain distribuida
compuesta por un Coordinador, Pool Manager, CPU Scaler y múltiples workers (CPU y
GPU), desplegada sobre Kubernetes en Google Cloud.

El sistema permite procesar transacciones, generar bloques mediante un
mecanismo de Prueba de Trabajo (PoW) y escalar dinámicamente según la
carga del sistema.

## Arquitectura General

La solución está compuesta por:

-   **Coordinador**: Construye bloques, valida soluciones y mantiene el
    estado de la blockchain. Provee una interfaz web para el uso del sistema.
-   **Pool Manager**: Distribuye bloques a los workers y mantiene registro de estos.
-   **Workers CPU**: Instancias desplegadas en Google Cloud.
-   **Worker GPU**: Ejecutado en hardware externo.
-   **CPU Scaler**: Escala CPUs dependiendo de los recursos disponibles.
-   **RabbitMQ**: Cola de mensajes para distribución de tareas.
-   **Redis**: Coordinación distribuida y manejo de locks.
-   **Prometheus + Grafana**: Monitoreo y métricas.
-   **NGINX Ingress**: Exposición de servicios.
-   **Terraform**: Aprovisionamiento de infraestructura.
-   **GitHub Actions**: Pipelines de despliegue.


##  Monitoreo

El sistema expone métricas en Prometheus y dashboards en Grafana,
incluyendo:

-   Throughput (bloques/segundo)
-   Hash rate por nodo
-   Latencia total del bloque
-   Intentos promedio por bloque
-   Escalabilidad bajo carga


##  Objetivo del Proyecto

Aplicar conceptos de:

-   Sistemas distribuidos
-   Tolerancia a fallos
-   Escalabilidad horizontal
-   Autoscaling en Kubernetes
-   Infraestructura como código
-   Monitoreo y observabilidad


##  Limitaciones

Actualmente no se implementa cifrado TLS interno ni externo y los
workers externos se conectan mediante endpoints públicos. Se proponen
mejoras futuras como integración mediante VPC o VPN privada y uso de Managed
Instance Groups.

Para este proyecto se utilizó hardware local para pruebas con el Worker GPU y se puede ejecutar con Docker Compose si se tiene el hardware compatible y [CUDA ToolKit](https://developer.nvidia.com/cuda/toolkit) instalado. Para un entorno productivo real simplemente se puede mudar la imagen Docker a una VM que soporte este tipo de Hardware.

## Links

Una vez desplegado el proyecto a traves de los pipelines, se puede hacer uso del sistema mediante estos links en conjunto con el **INGRESS_IP** y las credenciales. Se proveen los links del actual despliegue:

- [Dashboard del Coordinador](http://coordinator.136.119.226.21.nip.io)
- [Grafana](http://grafana.136.119.226.21.nip.io)
- [Prometheus](http://prometheus.136.119.226.21.nip.io)

 ### Estado del coordinador: ![Estado](https://img.shields.io/website?url=http://coordinator.136.119.226.21.nip.io/status)

## Documentación

### Diagrama de alto nivel
![Diagrama de alto nive](diagramas/DiagramaAltoNivel.png)

### Diagrama de bajo nivel
![Diagrama de alto nive](diagramas/DiagramaBajoNivel.png)

### [Informe Final](https://docs.google.com/document/d/1EiJcK0CmU8JtHDLXZTYatXBYHDUkEhaPS4ySJ7yEGuk/edit?usp=sharing)

### [Consigna](https://drive.google.com/file/d/1MRHyacNmx8XRIm6SIeHQY_1l-WEYR-Dx/view?usp=sharing)

##  Tecnologías utilizadas

## 🚀 Tecnologías utilizadas

### Backend y Minería
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white)](https://gunicorn.org/)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

---

### Mensajería y Persistencia
[![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-FF6600?style=for-the-badge&logo=rabbitmq&logoColor=white)](https://www.rabbitmq.com/)

---

### Infraestructura y Orquestación
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![Google Kubernetes Engine](https://img.shields.io/badge/GKE-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com/kubernetes-engine)
[![Helm](https://img.shields.io/badge/Helm-0F1689?style=for-the-badge&logo=helm&logoColor=white)](https://helm.sh/)
[![NGINX](https://img.shields.io/badge/NGINX-009639?style=for-the-badge&logo=nginx&logoColor=white)](https://nginx.org/)
[![Terraform](https://img.shields.io/badge/Terraform-7B42BC?style=for-the-badge&logo=terraform&logoColor=white)](https://www.terraform.io/)

---

### CI/CD
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)

---

### Monitoreo y Observabilidad
[![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)](https://grafana.com/)
[![Loki](https://img.shields.io/badge/Loki-000000?style=for-the-badge&logo=grafana&logoColor=white)](https://grafana.com/oss/loki/)
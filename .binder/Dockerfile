################################################################################
### Minimal Binder Dockerfile with FEniCS 2019                               ###
### École Polytechnique, Palaiseau, France                                   ###
################################################################################

# Partir de l'image officielle FEniCS 2019
FROM quay.io/fenicsproject/stable:2019.1.0

# Installer JupyterLab et matplotlib
RUN pip install --no-cache-dir jupyterlab matplotlib

# Créer l'utilisateur jovyan comme attendu par Binder
ARG NB_USER=jovyan
ARG NB_UID=1000
RUN useradd -m -s /bin/bash -N -u ${NB_UID} ${NB_USER}

# Passer à l'utilisateur jovyan
USER ${NB_USER}
ENV HOME /home/${NB_USER}
WORKDIR ${HOME}

# Copier le repo dans le conteneur
COPY --chown=${NB_UID} . ${HOME}
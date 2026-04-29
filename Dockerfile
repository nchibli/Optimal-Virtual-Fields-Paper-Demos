FROM quay.io/fenicsproject/stable:latest

USER root

RUN pip install numpy matplotlib scipy

USER ${NB_USER}



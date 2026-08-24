ARG HW_TAG=latest
FROM audrey16/principles-of-autonomy:${HW_TAG}

RUN echo "Running with homework tag: ${HW_TAG}"

COPY jupyter_server_config.py /etc/jupyter/

# Expose the Jupyter Lab port
EXPOSE 9000

ARG HW_DIR
ENV HW_DIR=${HW_DIR}
WORKDIR /work/homeworks/${HW_DIR}

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=9000", "--no-browser", "--allow-root"]
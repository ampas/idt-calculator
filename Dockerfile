FROM python:3.12
ENV PYTHON_MAJOR_MINOR_VERSION 3.12

RUN apt-get update \
    && apt-get -y install build-essential gcc g++ build-essential git cmake mlocate automake autotools-dev libtool  \
    libltdl-dev autoconf pkg-config m4 tree ffmpeg libsm6 libxext6 libpng-dev libgif-dev libwebp-dev libraw-dev \
    libopenjp2-7-dev libopenexr-dev libdcmtk-dev libtiff-dev libtbb-dev libfreetype6-dev libboost-all-dev \
    libopencv-dev pybind11-dev

ENV BUILD_DIR=/oiio_build \
    OIIO_VER=2.4.17.0 \
    OCIO_VER=v2.3.2 \
    PACKAGE_ROOT=/oiio_build/install \
    PYBIND11_VERSION=v2.11.1

# Create the build directory for OIIO
RUN mkdir -p "${BUILD_DIR}"

# Download and extract the OpenImageIO source
WORKDIR ${BUILD_DIR}
RUN curl -LO https://github.com/OpenImageIO/oiio/archive/refs/tags/v${OIIO_VER}.tar.gz \
    && tar xzf v${OIIO_VER}.tar.gz

# Set the OpenColorIO version environment variable and build OpenImageIO
WORKDIR ${BUILD_DIR}/OpenImageIO-${OIIO_VER}
RUN export OPENCOLORIO_VERSION=${OCIO_VER} \
    && src/build-scripts/build_pybind11.bash \
    && src/build-scripts/build_opencolorio.bash \
    && OIIO_BUILD_TESTS=0 OIIO_BUILD_TOOLS=0 make BUILD_SHARED_LIBS=0 CMAKE_CXX_STANDARD=14 PYBIND11_VERSION=v2.11.1 ENABLE_DICOM=0 ENABLE_DCMTK=0 \
    && make install

# Prepare the package
RUN mkdir -p "${PACKAGE_ROOT}/usr" \
    && cp -r "${BUILD_DIR}/OpenImageIO-${OIIO_VER}/dist/." "${PACKAGE_ROOT}/" \
    && cp -r "${BUILD_DIR}/OpenImageIO-${OIIO_VER}/ext/dist/." "${PACKAGE_ROOT}/"

#Set The PYTHON PATH
ENV PYTHONPATH="${PYTHONPATH}:/oiio_build/install/lib/python${PYTHON_MAJOR_MINOR_VERSION}/site-packages/OpenImageIO"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/oiio_build/install/lib/"

# Continue With IDT Calculator
WORKDIR /tmp
COPY ./requirements.txt /tmp
RUN pip install -r requirements.txt \
    && rm /tmp/requirements.txt

ARG CACHE_DATE

RUN mkdir -p /home/ampas/idt-calculator
WORKDIR /home/ampas/idt-calculator
COPY . /home/ampas/idt-calculator

CMD sh -c 'if [ -z "${SSL_CERTIFICATE}" ]; then \
    gunicorn --timeout 1200 --log-level debug -b 0.0.0.0:8000 index:SERVER; else \
    gunicorn --timeout 1200 --certfile "${SSL_CERTIFICATE}" --keyfile "${SSL_KEY}" --log-level debug -b 0.0.0.0:8000 index:SERVER; fi'

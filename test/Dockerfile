# This dockerfile is for continuous integration of the mycroft-precise repostiory

# Build an environment that can run the Precise wake word spotter.
FROM python:3.7-slim as precise-build
ENV TERM linux
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -y install git python3-scipy cython libhdf5-dev python3-h5py portaudio19-dev swig libpulse-dev libatlas-base-dev
RUN mkdir -p /root/allure /opt/mycroft/mycroft-precise /root/code-quality
WORKDIR /opt/mycroft
COPY requirements/test.txt mycroft-precise/requirements/
RUN pip install -r mycroft-precise/requirements/test.txt
COPY requirements/prod.txt mycroft-precise/requirements/
RUN pip install -r mycroft-precise/requirements/prod.txt
COPY . mycroft-precise

# Clone the devops repository, which contiains helper scripts for some continuous
# integraion tasks. Run the code_check.py script which performs linting (using PyLint)
# and code formatting (using Black)
FROM precise-build as code-checker
ARG github_api_key
ENV GITHUB_API_KEY=$github_api_key
RUN pip install pipenv
RUN git clone https://$github_api_key@github.com/MycroftAI/devops.git
WORKDIR /opt/mycroft/devops/jenkins
RUN pipenv install
ENTRYPOINT ["pipenv", "run", "python","-m", "pipeline.code_check", "--repository", "mycroft-precise", "--pull-request", "PR-149"]

# Run the tests defined in the precise repository
FROM precise-build as test-runner
WORKDIR /opt/mycroft/mycroft-precise
ENTRYPOINT ["pytest"]

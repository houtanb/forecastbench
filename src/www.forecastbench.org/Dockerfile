FROM ruby:3.2

WORKDIR /srv/jekyll
COPY Gemfile Gemfile.lock ./

RUN apt-get update && \
    apt-get install -y ruby-full build-essential zlib1g-dev && \
    gem install jekyll bundler minimal-mistakes-jekyll && \
    bundle install && \
    apt-get install -y apt-transport-https ca-certificates gnupg curl git openssh-client lsb-release && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk dumb-init

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

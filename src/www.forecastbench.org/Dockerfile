FROM ruby:3.2

WORKDIR /srv/jekyll
COPY Gemfile Gemfile.lock ./

RUN apt-get update && apt-get install -y ruby-full build-essential zlib1g-dev && \
    gem install jekyll bundler minimal-mistakes-jekyll && \
    bundle install

COPY . .

ENV MOUNTED_BUCKET=/mnt/forecastbench-cloud_run_job_workspace_scratch-dev

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

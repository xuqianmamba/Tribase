docker build -t panjd123/tribase-env:latest .

docker run -d \
  --user "$(id -u):$(id -g)" \
  --name tribase-dev \
  -v .:/app/tribase \
  --restart always \
  panjd123/tribase-env \
  tail -f /dev/null
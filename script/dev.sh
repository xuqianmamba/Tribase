docker build -t panjd123/tribase-env:latest .

docker run -d \
  --user 1000:1000 \
  --name tribase-dev \
  -v .:/app/tribase \
  --restart always \
  panjd123/tribase-env \
  tail -f /dev/null
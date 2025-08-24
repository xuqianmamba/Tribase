docker pull panjd123/tribase-env:latest
docker run -d \
  --user 1000:1000 \
  --name tribase-dev \
  -v .:/app/tribase \
  --restart always \
  panjd123/tribase-env \
  tail -f /dev/null
pipx install gdown
gdown https://drive.google.com/file/d/12wFLDNStJU02pEn7VcAs00LyS7uzcAbl/view?usp=sharing --fuzzy
unzip -o benchmarks.zip
docker exec -it tribase-dev bash script/build.sh
docker exec -it tribase-dev bash figures/run.sh
docker exec -it tribase-dev bash figures/draw.sh

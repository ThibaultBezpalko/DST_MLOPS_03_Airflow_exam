# Préparer la VM
 Si l'interface Airflow n'est pas fonctionnelle, c'est à cause des résidus du premier docker-compose.yml. On peut alors :
 - réinitialiser la VM
 - docker system prune -a

# Préparer l'environnement avant de réaliser l'examen
## shutting down previous containers
docker-compose down 

## deleting previous docker-compose
rm docker-compose.yaml

## downloading new docker-compose.yml file
wget https://dst-de.s3.eu-west-3.amazonaws.com/airflow_fr/eval/docker-compose.yaml

## creating directories
mkdir ./dags ./logs ./plugins
mkdir clean_data
mkdir raw_files

## if you have permission problems
sudo chmod -R 777 logs/
sudo chmod -R 777 dags/
sudo chmod -R 777 plugins/

echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env

docker-compose up airflow-init

wget https://dst-de.s3.eu-west-3.amazonaws.com/airflow_avance_fr/eval/data.csv -O clean_data/data.csv
echo '[]' >> raw_files/null_file.json

## starting docker-compose
docker-compose up -d
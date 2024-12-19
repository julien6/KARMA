from flask import Flask, request, jsonify
import os
import time
import requests

app = Flask(__name__)

# Configuration des adresses des services en aval
DOWNSTREAM_SERVICES = os.getenv("DOWNSTREAM_SERVICES", "").split(",")

# Configuration du throughput et du délai de traitement
THROUGHPUT = int(os.getenv("THROUGHPUT", 80))  # En données par seconde
PROCESSING_TIME = 1 / THROUGHPUT  # Temps simulé pour chaque unité de donnée

def consume_resources(cpu_percent, ram_mb):
    """ Simule la consommation de CPU et de RAM """
    busy_time = cpu_percent / 100
    idle_time = 1 - busy_time
    start_time = time.time()
    while time.time() - start_time < busy_time:
        pass  # Boucle active pour consommer du CPU
    time.sleep(idle_time)

    data = []
    try:
        for _ in range(int(ram_mb * 1024 / 64)):
            data.append(bytearray(64 * 1024))
        time.sleep(1)
    except MemoryError:
        print("Warning: Memory allocation failed. System may not have enough RAM.")
    finally:
        del data

@app.route("/process", methods=["POST"])
def process():
    """ Simule la réception, le traitement et la transmission des données """
    data = request.json
    input_data = data.get("input", "0000")  # Données par défaut si rien n'est passé
    print(f"Received data: {input_data}")

    # Simuler le traitement
    size = len(input_data)
    time.sleep(size * PROCESSING_TIME)
    consume_resources(cpu_percent=40.0, ram_mb=512)

    # Envoyer des requêtes aux services en aval
    responses = []
    for service in DOWNSTREAM_SERVICES:
        if service:
            try:
                response = requests.post(
                    f"http://{service}:5000/process",
                    json={"input": input_data}
                )
                responses.append({"service": service, "status": response.status_code})
            except Exception as e:
                responses.append({"service": service, "error": str(e)})

    print(f"Processed data sent downstream: {responses}")
    return jsonify({"status": "processed", "responses": responses}), 200

if __name__ == "__main__":
    print("Starting mock service for b...")
    app.run(host="0.0.0.0", port=5000)

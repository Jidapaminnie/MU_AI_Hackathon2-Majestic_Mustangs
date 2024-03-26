# Siriraj Helpdesk Chatbot by Majestic Mustangs team
### Project for MU AI Hackathon 2 at Faculty of ICT, Mahidol during March 23-24, 2024.

Description: A chatbot for answering questions concerning Siriraj doctors and clinics as well as scheduling hospital visits.
This chatbot was built using Vertex AI models.

### Overview

- Data extraction using web scraping
- Index creation and data querying using Llama Index and Langchain
- Web service development using Flask
- Web deployment and UI development using React.js (See [Front-end section](#frontend))

## Scrape data

See `readme.md` the folder `scripts`.

## Running the webservice

### Prerequisites

#### Create Index directories

Ran every cell in the `tutorial_notebooks/index_creation.ipynb` to create indices.

#### .env file

Create a `.env` file in the folder `webserver`. The file content must consist of the following

```
SERVICE_ACCOUNT_PATH="<path to GCP credential with vertex AI admin permission>"
MERGING_INDEX_DIR="./tutorial_notebooks/merging_index"
CLINIC_INDEX_DIR="./tutorial_notebooks/clinic_index"
CLINIC_DOCTOR_INDEX_DIR="./tutorial_notebooks/clinic_doctor_index"
PORT=8876
```

This `.env` file is used by the webservice as settings. All of them except PORT specify the location of index directory or gcp credential.

#### Installing python and its packages

The authors used python 3.11.7 on Windows 11. The python environment used is in the `environment.yml` which can be install easily with `conda`.

```bash
conda env create -f environment.yml # installing env
conda activate llama_env # using env
```

However, we have one of our teammate with MacOS unable to install this python env. We guessed that some library need `pywin32` which cannot be install else where beside Windows (since it is Windows APIs) and we did not have time to fix this, so we leave it as is. Good luck hunting this bug 🤦‍♂️🤦‍♀️.

### Running the webservice

After activate the python environment mentioned in the previous section. The command below will start the webservice on port 8876 (change this on `.env` file). 

```bash
python webserver/app.py
```

### POST /chatquery

Send a HTTP POST request to the `/chatquery` to start chatting. The request body must be a json that consist of
```json
{
    "id": "id of message, we use uuid4 but it can be any",
    "timestamp": "datetime in ISO",
    "text": "text to be chat with the chatbot",
    "sender": "name to identify sender. keep name the same and the bot will can memorize who it just talk with."
}
```

It will return json of the following 
```json
{
    "id": "id of message, we use uuid4 but it can be any",
    "timestamp": "datetime in ISO",
    "text": "text that chat bot answer",
    "sender": "chat bot" // we fixed this
}
```

## Frontend

We also develop a front end UI that is meant to be integrated with this repository. See
[github.com/makorn645/muai-2024-majestic-mustangs-frontend](https://github.com/makorn645/muai-2024-majestic-mustangs-frontend).

## Authors

The member of Majestic Mustangs

- [Danaidech Ardsamai](https://github.com/nungsorb)
- [Jidapa Chaocanapricha](https://github.com/Jidapaminnie) 
- [Makornthawat E Emery](https://github.com/makorn645)
- [Napahatai Sittirit](https://github.com/pinglarin)
- [Phuriwat Angkoondittaphong](https://github.com/ACitronella)


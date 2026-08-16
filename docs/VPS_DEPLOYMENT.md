# AkibaAI Contabo VPS demo deployment

This deployment runs the current Streamlit application and its reviewed XGBoost
model in Docker. SQLite is stored in a named Docker volume, so assessments remain
available after a container restart, rebuild, or `docker compose down`.

This is a demo configuration. Use synthetic data only. It does not add user
accounts, TLS, database encryption, or multi-user authorization. The access key
protects only the Settings data-management controls.

## 1. Prepare the VPS

Use a current Ubuntu or Debian VPS. Install Docker Engine and the Docker Compose
plugin using Docker's official instructions, then confirm both are available:

```bash
docker --version
docker compose version
```

Clone the repository and check out the branch or release containing this setup:

```bash
git clone https://github.com/SKitavi/akiba-ai.git
cd akiba-ai
git checkout feature/streamlit-ui
```

The branch must first be pushed to GitHub, or merged into the branch you deploy.

## 2. Configure the demo

Create the local environment file:

```bash
cp .env.example .env
nano .env
```

For the requested demo, keep:

```dotenv
SETTINGS_ACCESS_KEY=CMU#AB39
```

Because `#` is part of the value, do not add an inline comment to that line.
The `.env` file is excluded from Git. Change the key there later without editing
application code.

## 3. Start the application

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 akiba-ai
```

If an older image reports `ModuleNotFoundError: No module named 'src'`, pull the
latest commit and force Compose to recreate the application container:

```bash
git pull
docker compose up -d --build --force-recreate
```

If UFW is enabled, allow the demo port:

```bash
sudo ufw allow 8501/tcp
```

Open `http://YOUR_VPS_IP:8501`. In **Settings**, enter `CMU#AB39` to unlock the
demo-data and reset controls for that browser session. Use **Lock settings** when
finished.

## 4. Update without losing SQLite data

```bash
git pull
docker compose up -d --build
```

The `akiba_data` named volume remains attached. `docker compose down` also
preserves it. Do not run `docker compose down -v` unless you intentionally want
to delete the database volume.

## 5. Back up SQLite

Create a consistent SQLite backup inside the running container, then copy it to
the VPS working directory:

```bash
docker compose exec akiba-ai python -c "import sqlite3; source=sqlite3.connect('/app/runtime/akiba_ai.db'); backup=sqlite3.connect('/app/runtime/akiba_ai.backup.db'); source.backup(backup); backup.close(); source.close()"
docker compose cp akiba-ai:/app/runtime/akiba_ai.backup.db ./akiba_ai.backup.db
```

Store the copied file somewhere protected. It contains all persisted assessment
records.

## 6. Operations

```bash
# Follow logs
docker compose logs -f akiba-ai

# Restart the app
docker compose restart akiba-ai

# Stop while preserving data
docker compose down
```

For a public-facing or longer-lived deployment, put the application behind an
HTTPS reverse proxy, restrict port 8501 at the firewall, replace the shared key
with real authentication, and move concurrent production data to PostgreSQL.

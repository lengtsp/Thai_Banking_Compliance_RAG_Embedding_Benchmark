#!/bin/bash
# Setup PostgreSQL database with pgvector on port 5432
# Run: sudo bash setup_db.sh

set -e

PG_PORT=5432
DB_NAME=test_embedding
DB_USER=myuser
DB_PASS=mypassword

echo "🔧 Setting up database on PostgreSQL port $PG_PORT..."

su - postgres -c "psql -p $PG_PORT" <<SQL
-- Create user (ignore error if exists)
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$DB_USER') THEN
    CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';
    RAISE NOTICE 'User $DB_USER created';
  ELSE
    RAISE NOTICE 'User $DB_USER already exists';
  END IF;
END
\$\$;

-- Create database (ignore error if exists)
SELECT 'CREATE DATABASE $DB_NAME OWNER $DB_USER'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$DB_NAME')
\gexec

GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
SQL

echo "✅ User and database ready"

# Enable pgvector extension inside the database
su - postgres -c "psql -p $PG_PORT -d $DB_NAME" <<SQL
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
SQL

echo ""
echo "✅ pgvector extension enabled in $DB_NAME"
echo ""
echo "👉 Now update .env:  DB_PORT=$PG_PORT"
echo "👉 Then run:         python app.py"

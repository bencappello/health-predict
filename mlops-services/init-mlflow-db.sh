#!/bin/bash
set -e

echo "Attempting to create database mlflowdb..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "CREATE DATABASE mlflowdb;"
echo "Database creation command executed." 
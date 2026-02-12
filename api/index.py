# api/index.py
from app import app

# Vercel serverless function handler
def handler(event, context):
    return app(event, context)
#!/usr/bin/env python3
"""
Database Setup Script for CivicLink
Run this to generate SQL schema for Supabase
"""

from database import create_tables_sql

def main():
    print("=" * 80)
    print("CivicLink Database Setup")
    print("=" * 80)
    print()
    print("Copy the SQL below and paste it into your Supabase SQL Editor:")
    print()
    print("=" * 80)
    print()
    print(create_tables_sql())
    print()
    print("=" * 80)
    print()
    print("After running the SQL:")
    print("1. Go to Supabase Dashboard → SQL Editor")
    print("2. Create a new query")
    print("3. Paste the SQL above")
    print("4. Click 'Run' to execute")
    print()
    print("This will create all necessary tables and initial data.")
    print("=" * 80)

if __name__ == "__main__":
    main()

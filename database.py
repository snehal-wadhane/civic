# database.py - Database connection and schema definitions
from supabase import create_client, Client
from config import settings
import logging
import os
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SECRET)
# url: str = os.environ.get("SUPABASE_URL")
# key: str = os.environ.get("SUPABASE_SECRET")
# supabase: Client = create_client(url, key)
# Database Schema Documentation
"""
Table: users
- uid (SERIAL PRIMARY KEY)
- email (VARCHAR UNIQUE NOT NULL)
- password_hash (VARCHAR NOT NULL)
- full_name (VARCHAR)
- phone (VARCHAR)
- role (VARCHAR DEFAULT 'citizen') -- citizen, admin, department_staff
- department_id (INTEGER REFERENCES departments(department_id))
- created_at (TIMESTAMP DEFAULT NOW())
- updated_at (TIMESTAMP DEFAULT NOW())

Table: departments
- department_id (SERIAL PRIMARY KEY)
- name (VARCHAR UNIQUE NOT NULL) -- Public Works, Electricity Board, etc.
- description (TEXT)
- email (VARCHAR)
- phone (VARCHAR)
- avg_resolution_hours (INTEGER DEFAULT 72)
- active (BOOLEAN DEFAULT TRUE)
- created_at (TIMESTAMP DEFAULT NOW())

Table: issue_types
- issue_type_id (SERIAL PRIMARY KEY)
- name (VARCHAR UNIQUE NOT NULL) -- pothole, streetlight, water_leak, etc.
- category (VARCHAR) -- Road, Electricity, Water, Sanitation
- department_id (INTEGER REFERENCES departments(department_id))
- default_priority (VARCHAR DEFAULT 'Medium') -- Low, Medium, High, Critical
- ml_model_path (VARCHAR) -- Path to specific ML model
- active (BOOLEAN DEFAULT TRUE)
- created_at (TIMESTAMP DEFAULT NOW())

Table: problems
- pid (SERIAL PRIMARY KEY)
- uid (INTEGER REFERENCES users(uid))
- email (VARCHAR)
- issue_type_id (INTEGER REFERENCES issue_types(issue_type_id))
- department_id (INTEGER REFERENCES departments(department_id))
-
- photo (VARCHAR) -- Supabase Storage URL
- Description (TEXT)
- Latitude (VARCHAR)
- Longitude (VARCHAR)
- address (TEXT)
-
- status (VARCHAR DEFAULT 'Reported') 
  -- Reported, Verified, Assigned, In Progress, Resolved, Closed, Rejected
- priority (VARCHAR DEFAULT 'Medium') -- Low, Medium, High, Critical
- 
- severity_status (FLOAT) -- ML calculated score
- overall_severity (VARCHAR) -- Low, Moderate, High
- confidence_score (FLOAT) -- ML confidence
- 
- assigned_to (INTEGER REFERENCES users(uid))
- verified_by (INTEGER REFERENCES users(uid))
- 
- estimated_resolution_hours (INTEGER)
- actual_resolution_hours (INTEGER)
-
- reported_at (TIMESTAMP DEFAULT NOW())
- verified_at (TIMESTAMP)
- assigned_at (TIMESTAMP)
- resolved_at (TIMESTAMP)
- closed_at (TIMESTAMP)
- updated_at (TIMESTAMP DEFAULT NOW())

Table: status_history
- history_id (SERIAL PRIMARY KEY)
- pid (INTEGER REFERENCES problems(pid))
- old_status (VARCHAR)
- new_status (VARCHAR)
- changed_by (INTEGER REFERENCES users(uid))
- notes (TEXT)
- created_at (TIMESTAMP DEFAULT NOW())

Table: feedback
- feedback_id (SERIAL PRIMARY KEY)
- pid (INTEGER REFERENCES problems(pid))
- uid (INTEGER REFERENCES users(uid))
- rating (INTEGER CHECK (rating >= 1 AND rating <= 5))
- comment (TEXT)
- satisfaction_level (VARCHAR) -- Very Satisfied, Satisfied, Neutral, Dissatisfied, Very Dissatisfied
- would_recommend (BOOLEAN)
- created_at (TIMESTAMP DEFAULT NOW())

Table: notifications
- notification_id (SERIAL PRIMARY KEY)
- uid (INTEGER REFERENCES users(uid))
- pid (INTEGER REFERENCES problems(pid))
- type (VARCHAR) -- status_update, assignment, resolution, feedback_request
- title (VARCHAR)
- message (TEXT)
- read (BOOLEAN DEFAULT FALSE)
- sent_via (VARCHAR) -- app, email, sms
- created_at (TIMESTAMP DEFAULT NOW())

Table: analytics
- analytics_id (SERIAL PRIMARY KEY)
- date (DATE)
- department_id (INTEGER REFERENCES departments(department_id))
- issue_type_id (INTEGER REFERENCES issue_types(issue_type_id))
- total_reported (INTEGER DEFAULT 0)
- total_resolved (INTEGER DEFAULT 0)
- avg_resolution_hours (FLOAT)
- high_priority_count (INTEGER DEFAULT 0)
- citizen_satisfaction_avg (FLOAT)
- created_at (TIMESTAMP DEFAULT NOW())
"""

# SQL Schema Creation Scripts
def create_tables_sql():
    """
    SQL scripts to create all tables.
    Run these in Supabase SQL Editor.
    """
    return """
    -- Enable UUID extension
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    
    -- Departments Table
    CREATE TABLE IF NOT EXISTS departments (
        department_id SERIAL PRIMARY KEY,
        name VARCHAR(100) UNIQUE NOT NULL,
        description TEXT,
        email VARCHAR(255),
        phone VARCHAR(20),
        avg_resolution_hours INTEGER DEFAULT 72,
        active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Insert default departments
    INSERT INTO departments (name, description, avg_resolution_hours) VALUES
    ('Public Works', 'Roads, potholes, and infrastructure', 48),
    ('Electricity Board', 'Streetlights and electrical issues', 24),
    ('Water Department', 'Water leaks and supply issues', 36),
    ('Sanitation', 'Garbage collection and cleanliness', 12),
    ('Traffic Management', 'Traffic signals and road safety', 24)
    ON CONFLICT (name) DO NOTHING;
    
    -- Issue Types Table
    CREATE TABLE IF NOT EXISTS issue_types (
        issue_type_id SERIAL PRIMARY KEY,
        name VARCHAR(100) UNIQUE NOT NULL,
        category VARCHAR(50),
        department_id INTEGER REFERENCES departments(department_id),
        default_priority VARCHAR(20) DEFAULT 'Medium',
        ml_model_path VARCHAR(255),
        active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Insert default issue types
    INSERT INTO issue_types (name, category, department_id, default_priority) VALUES
    ('Pothole', 'Road', 1, 'High'),
    ('Road Damage', 'Road', 1, 'High'),
    ('Broken Streetlight', 'Electricity', 2, 'Medium'),
    ('Water Leak', 'Water', 3, 'High'),
    ('Sewage Overflow', 'Water', 3, 'Critical'),
    ('Garbage Accumulation', 'Sanitation', 4, 'Medium'),
    ('Blocked Drain', 'Water', 3, 'High'),
    ('Traffic Signal Malfunction', 'Traffic', 5, 'Critical'),
    ('Illegal Dumping', 'Sanitation', 4, 'Low'),
    ('Tree Fallen', 'Road', 1, 'High')
    ON CONFLICT (name) DO NOTHING;
    
    -- Users Table (Enhanced)
    CREATE TABLE IF NOT EXISTS users (
        uid SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        full_name VARCHAR(100),
        phone VARCHAR(20),
        role VARCHAR(20) DEFAULT 'citizen',
        department_id INTEGER REFERENCES departments(department_id),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Problems Table (Enhanced)
    CREATE TABLE IF NOT EXISTS problems (
        pid SERIAL PRIMARY KEY,
        uid INTEGER REFERENCES users(uid),
        email VARCHAR(255),
        issue_type_id INTEGER REFERENCES issue_types(issue_type_id),
        department_id INTEGER REFERENCES departments(department_id),
        
        photo VARCHAR(500),
        Description TEXT,
        Latitude VARCHAR(50),
        Longitude VARCHAR(50),
        address TEXT,
        
        status VARCHAR(50) DEFAULT 'Reported',
        priority VARCHAR(20) DEFAULT 'Medium',
        
        severity_status FLOAT,
        overall_severity VARCHAR(20),
        confidence_score FLOAT,
        
        assigned_to INTEGER REFERENCES users(uid),
        verified_by INTEGER REFERENCES users(uid),
        
        estimated_resolution_hours INTEGER,
        actual_resolution_hours INTEGER,
        
        reported_at TIMESTAMP DEFAULT NOW(),
        verified_at TIMESTAMP,
        assigned_at TIMESTAMP,
        resolved_at TIMESTAMP,
        closed_at TIMESTAMP,
        updated_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Status History Table
    CREATE TABLE IF NOT EXISTS status_history (
        history_id SERIAL PRIMARY KEY,
        pid INTEGER REFERENCES problems(pid) ON DELETE CASCADE,
        old_status VARCHAR(50),
        new_status VARCHAR(50),
        changed_by INTEGER REFERENCES users(uid),
        notes TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Feedback Table
    CREATE TABLE IF NOT EXISTS feedback (
        feedback_id SERIAL PRIMARY KEY,
        pid INTEGER REFERENCES problems(pid) ON DELETE CASCADE,
        uid INTEGER REFERENCES users(uid),
        rating INTEGER CHECK (rating >= 1 AND rating <= 5),
        comment TEXT,
        satisfaction_level VARCHAR(50),
        would_recommend BOOLEAN,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Notifications Table
    CREATE TABLE IF NOT EXISTS notifications (
        notification_id SERIAL PRIMARY KEY,
        uid INTEGER REFERENCES users(uid) ON DELETE CASCADE,
        pid INTEGER REFERENCES problems(pid) ON DELETE CASCADE,
        type VARCHAR(50),
        title VARCHAR(255),
        message TEXT,
        read BOOLEAN DEFAULT FALSE,
        sent_via VARCHAR(20),
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Analytics Table
    CREATE TABLE IF NOT EXISTS analytics (
        analytics_id SERIAL PRIMARY KEY,
        date DATE,
        department_id INTEGER REFERENCES departments(department_id),
        issue_type_id INTEGER REFERENCES issue_types(issue_type_id),
        total_reported INTEGER DEFAULT 0,
        total_resolved INTEGER DEFAULT 0,
        avg_resolution_hours FLOAT,
        high_priority_count INTEGER DEFAULT 0,
        citizen_satisfaction_avg FLOAT,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(date, department_id, issue_type_id)
    );
    
    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_problems_status ON problems(status);
    CREATE INDEX IF NOT EXISTS idx_problems_department ON problems(department_id);
    CREATE INDEX IF NOT EXISTS idx_problems_priority ON problems(priority);
    CREATE INDEX IF NOT EXISTS idx_problems_location ON problems(Latitude, Longitude);
    CREATE INDEX IF NOT EXISTS idx_problems_assigned ON problems(assigned_to);
    CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(uid, read);
    
    -- Create trigger for updated_at
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    
    CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
    CREATE TRIGGER update_problems_updated_at BEFORE UPDATE ON problems
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """

# Helper function to check if tables exist
def check_database_setup():
    """Check if database is properly set up"""
    try:
        # Try to fetch from departments table
        result = supabase.table("departments").select("*").limit(1).execute()
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database setup check failed: {e}")
        logger.info("Please run the SQL scripts in Supabase SQL Editor")
        return False

# Initialize database check
if __name__ == "__main__":
    print("Database Schema SQL:")
    print(create_tables_sql())
    print("\n" + "="*80)
    print("Copy the above SQL and run it in your Supabase SQL Editor")
    print("="*80)

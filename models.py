# models.py - Pydantic models and schemas
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

# Enums
class IssueStatus(str, Enum):
    REPORTED = "Reported"
    VERIFIED = "Verified"
    ASSIGNED = "Assigned"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    CLOSED = "Closed"
    REJECTED = "Rejected"

class Priority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class UserRole(str, Enum):
    CITIZEN = "citizen"
    DEPARTMENT_STAFF = "department_staff"
    ADMIN = "admin"

class SatisfactionLevel(str, Enum):
    VERY_SATISFIED = "Very Satisfied"
    SATISFIED = "Satisfied"
    NEUTRAL = "Neutral"
    DISSATISFIED = "Dissatisfied"
    VERY_DISSATISFIED = "Very Dissatisfied"

# User Schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    phone: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    uid: int
    email: str
    full_name: Optional[str]
    phone: Optional[str]
    role: str
    department_id: Optional[int]
    created_at: datetime

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None

# Department Schemas
class DepartmentResponse(BaseModel):
    department_id: int
    name: str
    description: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    avg_resolution_hours: int
    active: bool

# Issue Type Schemas
class IssueTypeResponse(BaseModel):
    issue_type_id: int
    name: str
    category: str
    department_id: int
    default_priority: str
    active: bool

# Problem Schemas
class ProblemCreate(BaseModel):
    issue_type_id: int
    description: str
    latitude: str
    longitude: str
    address: Optional[str] = None

class ProblemResponse(BaseModel):
    pid: int
    uid: int
    email: str
    issue_type_id: Optional[int]
    issue_type_name: Optional[str]
    department_id: Optional[int]
    department_name: Optional[str]
    
    photo: Optional[str]
    Description: Optional[str]
    Latitude: Optional[str]
    Longitude: Optional[str]
    address: Optional[str]
    
    status: str
    priority: str
    
    severity_status: Optional[float]
    overall_severity: Optional[str]
    confidence_score: Optional[float]
    
    assigned_to: Optional[int]
    assigned_to_name: Optional[str]
    verified_by: Optional[int]
    
    estimated_resolution_hours: Optional[int]
    actual_resolution_hours: Optional[int]
    
    reported_at: datetime
    verified_at: Optional[datetime]
    assigned_at: Optional[datetime]
    resolved_at: Optional[datetime]
    closed_at: Optional[datetime]
    updated_at: datetime

class ProblemUpdate(BaseModel):
    status: Optional[IssueStatus] = None
    priority: Optional[Priority] = None
    assigned_to: Optional[int] = None
    notes: Optional[str] = None

# Feedback Schemas
class FeedbackCreate(BaseModel):
    pid: int
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
    satisfaction_level: SatisfactionLevel
    would_recommend: bool

class FeedbackResponse(BaseModel):
    feedback_id: int
    pid: int
    uid: int
    rating: int
    comment: Optional[str]
    satisfaction_level: str
    would_recommend: bool
    created_at: datetime

# Notification Schemas
class NotificationResponse(BaseModel):
    notification_id: int
    uid: int
    pid: Optional[int]
    type: str
    title: str
    message: str
    read: bool
    created_at: datetime

# Analytics Schemas
class DepartmentStats(BaseModel):
    department_id: int
    department_name: str
    total_reported: int
    total_resolved: int
    in_progress: int
    avg_resolution_hours: float
    resolution_rate: float

class IssueTypeStats(BaseModel):
    issue_type_id: int
    issue_type_name: str
    count: int
    percentage: float

class DashboardStats(BaseModel):
    total_issues: int
    resolved_issues: int
    pending_issues: int
    resolution_rate: float
    avg_resolution_time: float
    high_priority_count: int
    by_department: List[DepartmentStats]
    by_issue_type: List[IssueTypeStats]
    recent_issues: List[ProblemResponse]

# Status History Schema
class StatusHistoryResponse(BaseModel):
    history_id: int
    pid: int
    old_status: Optional[str]
    new_status: str
    changed_by: Optional[int]
    changed_by_name: Optional[str]
    notes: Optional[str]
    created_at: datetime

# ML Prediction Schema
class MLPrediction(BaseModel):
    issue_type: str
    severity: str
    confidence: float
    detected_objects: List[dict]

# Resolution Time Estimate
class ResolutionEstimate(BaseModel):
    estimated_hours: int
    estimated_completion_date: datetime
    confidence: str  # High, Medium, Low
    factors: List[str]
from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str | None = None

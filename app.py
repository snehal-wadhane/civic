# app.py - Enhanced CivicLink API Application
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from typing import Optional, List
from datetime import datetime, timedelta
import logging
import uuid
import base64
import requests
import numpy as np

# Local imports
from config import settings
from database import supabase, check_database_setup
from auth import (
    get_password_hash, authenticate_user, create_access_token,
    get_current_user, get_current_active_user, get_admin_user, get_department_staff
)
from models import (
    UserCreate, UserLogin, UserResponse, Token,
    ProblemResponse, ProblemUpdate, FeedbackCreate, FeedbackResponse,
    NotificationResponse, DashboardStats, StatusHistoryResponse,
    IssueTypeResponse, DepartmentResponse, ResolutionEstimate
)
from ml_services import analyze_issue_image
from services import (
    haversine_distance, calculate_severity_score, determine_priority,
    auto_route_to_department, estimate_resolution_time,
    update_problem_status, auto_assign_to_staff,
    notify_status_change, request_feedback
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Powered Citizen Grievance Redressal System"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# STARTUP EVENT
# =====================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Check database connection
    if check_database_setup():
        logger.info("Database connection successful")
    else:
        logger.warning("Database setup incomplete - please run SQL scripts")

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/", tags=["Health"])
def root():
    """API health check"""
    return {
        "status": "online",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "timestamp": datetime.utcnow().isoformat()
    }

# =====================================================
# AUTHENTICATION ENDPOINTS
# =====================================================

@app.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register_user(user: UserCreate):
    """
    Register new user
    
    - **email**: Valid email address
    - **password**: Minimum 6 characters
    - **full_name**: Optional full name
    - **phone**: Optional phone number
    """
    try:
        # Check if email already exists
        existing = supabase.table("users").select("email").eq("email", user.email).execute()
        if existing.data:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Hash password
        password_hash = get_password_hash(user.password)
        
        # Insert user
        response = supabase.table("users").insert({
            "email": user.email,
            "password_hash": password_hash,
            "full_name": user.full_name,
            "phone": user.phone,
            "role": "citizen"
        }).execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to create user")
        
        user_data = response.data[0]
        return UserResponse(**user_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login(user: UserLogin):
    """
    Login user and get access token
    
    - **email**: User email
    - **password**: User password
    
    Returns JWT access token for authenticated requests
    """
    authenticated_user = await authenticate_user(user.email, user.password)
    
    if not authenticated_user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    
    # Create access token
    access_token = create_access_token(
        data={
            "sub": authenticated_user.email,
            "uid": authenticated_user.uid,
            "role": authenticated_user.role
        }
    )
    
    return Token(access_token=access_token, token_type="bearer")

@app.post("/auth/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible token endpoint"""
    authenticated_user = await authenticate_user(form_data.username, form_data.password)
    
    if not authenticated_user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={
            "sub": authenticated_user.email,
            "uid": authenticated_user.uid,
            "role": authenticated_user.role
        }
    )
    
    return Token(access_token=access_token, token_type="bearer")

@app.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current authenticated user information"""
    response = supabase.table("users").select("*").eq("uid", current_user.uid).execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(**response.data[0])

# =====================================================
# PROBLEM REPORTING & MANAGEMENT
# =====================================================

@app.post("/problems/report", tags=["Problems"])
async def report_problem(
    background_tasks: BackgroundTasks,
    issue_type_id: Optional[int] = Form(None),
    description: str = Form(...),
    latitude: str = Form(...),
    longitude: str = Form(...),
    address: Optional[str] = Form(None),
    image: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """
    Report new civic issue
    
    - **issue_type_id**: Optional - Auto-detected if not provided
    - **description**: Issue description
    - **latitude**: GPS latitude
    - **longitude**: GPS longitude
    - **address**: Optional address
    - **image**: Issue photo
    
    Process:
    1. Upload image immediately
    2. Create problem record
    3. Background: ML analysis + severity calculation
    4. Background: Auto-route to department
    5. Background: Auto-assign to staff
    6. Return problem ID immediately
    """
    try:
        # Read image
        img_bytes = await image.read()
        file_ext = image.filename.split(".")[-1] if "." in image.filename else "jpg"
        
        # Upload to Supabase Storage
        file_name = f"{uuid.uuid4()}.{file_ext}"
        supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).upload(file_name, img_bytes)
        photo_url = supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).get_public_url(file_name)
        
        # Create initial problem record
        problem_data = {
            "uid": current_user.uid,
            "email": current_user.email,
            "issue_type_id": issue_type_id,
            "photo": photo_url,
            "description": description,
            "latitude": latitude,
            "longitude": longitude,
            "address": address,
            "status": "Reported",
            "priority": "Medium"
        }
        
        response = supabase.table("problems").insert(problem_data).execute()
        pid = response.data[0]["pid"]
        
        # Process in background
        background_tasks.add_task(
            process_problem_analysis,
            pid, img_bytes, latitude, longitude, issue_type_id
        )
        
        # Create notification
        background_tasks.add_task(
            notify_status_change,
            pid, "Reported"
        )
        
        return {
            "status": "success",
            "message": "Issue reported successfully",
            "pid": pid,
            "note": "Analysis in progress - severity will be updated shortly"
        }
        
    except Exception as e:
        logger.error(f"Error reporting problem: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def process_problem_analysis(
    pid: int,
    img_bytes: bytes,
    latitude: str,
    longitude: str,
    issue_type_id: Optional[int]
):
    """Background task: Analyze problem and update details"""
    try:
        lat = float(latitude)
        lon = float(longitude)
        
        # Get issue type name if provided
        issue_type_name = None
        if issue_type_id:
            issue_type = supabase.table("issue_types").select("name").eq("issue_type_id", issue_type_id).execute()
            if issue_type.data:
                issue_type_name = issue_type.data[0]["name"]
        
        # ML Analysis
        ml_result = analyze_issue_image(img_bytes, issue_type_name)
        
        # If issue type not provided, detect from ML
        if not issue_type_id and ml_result.get("detected"):
            detected_type = ml_result.get("detected_issue_type")
            # Try to find matching issue type
            type_response = supabase.table("issue_types").select("issue_type_id").eq("name", detected_type).execute()
            if type_response.data:
                issue_type_id = type_response.data[0]["issue_type_id"]
                issue_type_name = detected_type
        
        # Calculate severity
        severity_score, overall_severity = calculate_severity_score(
            ml_result, lat, lon, issue_type_name or "Unknown"
        )
        
        # Determine priority
        priority = determine_priority(severity_score, issue_type_name or "Unknown")
        
        # Auto-route to department
        department_id = auto_route_to_department(issue_type_id) if issue_type_id else None
        
        # Estimate resolution time
        estimated_hours, confidence = (72, "Medium")
        if department_id and issue_type_id:
            estimated_hours, confidence = estimate_resolution_time(
                issue_type_id, priority, department_id
            )
        
        # Update problem
        update_data = {
            "issue_type_id": issue_type_id,
            "department_id": department_id,
            "severity_status": severity_score,
            "overall_severity": overall_severity,
            "confidence_score": ml_result.get("confidence", 0.0),
            "priority": priority,
            "estimated_resolution_hours": estimated_hours
        }
        
        supabase.table("problems").update(update_data).eq("pid", pid).execute()
        
        # Auto-assign to staff
        if department_id:
            assigned_staff = auto_assign_to_staff(department_id)
            if assigned_staff:
                supabase.table("problems").update({
                    "assigned_to": assigned_staff,
                    "status": "Assigned",
                    "assigned_at": datetime.utcnow().isoformat()
                }).eq("pid", pid).execute()
                
                # Notify assigned staff
                notify_status_change(pid, "Assigned")
        
        logger.info(f"Problem {pid} analyzed: severity={severity_score}, priority={priority}")
        
    except Exception as e:
        logger.error(f"Error in problem analysis: {e}")

@app.get("/problems/nearby", tags=["Problems"])
async def get_nearby_problems(
    latitude: float = Query(...),
    longitude: float = Query(...),
    radius_km: float = Query(None),
    include_images: bool = Query(False),
    status: Optional[str] = Query(None)
):
    """
    Get problems near a location
    
    - **latitude**: Your latitude
    - **longitude**: Your longitude  
    - **radius_km**: Search radius (default: 0.5 km)
    - **include_images**: Include base64 images
    - **status**: Filter by status
    """
    if radius_km is None:
        radius_km = settings.NEARBY_RADIUS_KM
    
    try:
        # Fetch all problems
        query = supabase.table("problems").select("*, issue_types(name), departments(name)")
        
        if status:
            query = query.eq("status", status)
        
        response = query.execute()
        all_problems = response.data
        
        # Filter by distance
        nearby = []
        for problem in all_problems:
            try:
                p_lat = float(problem.get("Latitude", 0))
                p_lon = float(problem.get("Longitude", 0))
                distance = haversine_distance(latitude, longitude, p_lat, p_lon)
                
                if distance <= radius_km:
                    problem["distance_km"] = round(distance, 2)
                    
                    # Add issue type name
                    if problem.get("issue_types"):
                        problem["issue_type_name"] = problem["issue_types"]["name"]
                    
                    # Add department name
                    if problem.get("departments"):
                        problem["department_name"] = problem["departments"]["name"]
                    
                    # Include image if requested
                    if include_images and problem.get("photo"):
                        try:
                            r = requests.get(problem["photo"], timeout=5)
                            if r.status_code == 200:
                                problem["photo_base64"] = base64.b64encode(r.content).decode("utf-8")
                        except:
                            pass
                    
                    nearby.append(problem)
            except:
                continue
        
        # Sort by distance
        nearby.sort(key=lambda x: x.get("distance_km", float('inf')))
        
        return {"status": "success", "count": len(nearby), "data": nearby}
        
    except Exception as e:
        logger.error(f"Error fetching nearby problems: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/problems/{pid}", tags=["Problems"])
async def get_problem_by_id(pid: int):
    """Get detailed problem information by ID"""
    try:
        response = supabase.table("problems") \
            .select("""
                *,
                reporter:users!problems_uid_fkey(*),
                assigned_staff:users!problems_assigned_to_fkey(*),
                verifier:users!problems_verified_by_fkey(*),
                issue:issue_types(*),
                department:departments(*)
            """) \
            .eq("pid", pid) \
            .execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Problem not found")

        problem = response.data[0]

        return {
            "status": "success",
            "data": problem
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching problem: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/problems/user/mine", tags=["Problems"])
async def get_my_problems(
    status: Optional[str] = Query(None),
    current_user = Depends(get_current_user)
):
    """Get all problems reported by current user"""
    try:
        query = supabase.table("problems").select("""
            *,
            issue_types(name),
            departments(name)
        """).eq("uid", current_user.uid)
        
        if status:
            query = query.eq("status", status)
        
        response = query.order("reported_at", desc=True).execute()
        
        for problem in response.data:
            if problem.get("issue_types"):
                problem["issue_type_name"] = problem["issue_types"]["name"]
            if problem.get("departments"):
                problem["department_name"] = problem["departments"]["name"]
        
        return {"status": "success", "count": len(response.data), "data": response.data}
        
    except Exception as e:
        logger.error(f"Error fetching user problems: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/problems/{pid}/status", tags=["Problems"])
async def update_problem_status_endpoint(
    pid: int,
    new_status: str = Body(..., embed=True),
    notes: Optional[str] = Body(None, embed=True),
    current_user = Depends(get_department_staff)
):
    """
    Update problem status (Department staff/Admin only)
    
    Valid statuses: Reported, Verified, Assigned, In Progress, Resolved, Closed, Rejected
    """
    valid_statuses = ["Reported", "Verified", "Assigned", "In Progress", "Resolved", "Closed", "Rejected"]
    
    if new_status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
    
    success = update_problem_status(pid, new_status, current_user.uid, notes)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update status")
    
    # Send notification
    notify_status_change(pid, new_status)
    
    # Request feedback if resolved
    if new_status == "Resolved":
        request_feedback(pid)
    
    return {"status": "success", "message": f"Problem {pid} updated to {new_status}"}

@app.get("/problems/{pid}/history", response_model=List[StatusHistoryResponse], tags=["Problems"])
async def get_problem_history(pid: int):
    """Get status change history for a problem"""
    try:
        response = supabase.table("status_history").select("""
            *,
            users(full_name, email)
        """).eq("pid", pid).order("created_at", desc=False).execute()
        
        history = []
        for item in response.data:
            item["changed_by_name"] = item.get("users", {}).get("full_name") if item.get("users") else None
            history.append(StatusHistoryResponse(**item))
        
        return history
        
    except Exception as e:
        logger.error(f"Error fetching problem history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# FEEDBACK ENDPOINTS
# =====================================================

@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(
    feedback: FeedbackCreate,
    current_user = Depends(get_current_user)
):
    """
    Submit feedback for resolved issue
    
    - **pid**: Problem ID
    - **rating**: 1-5 stars
    - **comment**: Optional comment
    - **satisfaction_level**: Very Satisfied, Satisfied, Neutral, Dissatisfied, Very Dissatisfied
    - **would_recommend**: Would you recommend this service?
    """
    try:
        # Check if problem exists and is resolved
        problem = supabase.table("problems").select("status, uid").eq("pid", feedback.pid).execute()
        
        if not problem.data:
            raise HTTPException(status_code=404, detail="Problem not found")
        
        # Check if user owns this problem
        if problem.data[0]["uid"] != current_user.uid:
            raise HTTPException(status_code=403, detail="You can only provide feedback for your own issues")
        
        # Check if already submitted feedback
        existing = supabase.table("feedback").select("feedback_id").eq("pid", feedback.pid).eq("uid", current_user.uid).execute()
        
        if existing.data:
            raise HTTPException(status_code=400, detail="Feedback already submitted for this issue")
        
        # Insert feedback
        feedback_data = feedback.dict()
        feedback_data["uid"] = current_user.uid
        
        response = supabase.table("feedback").insert(feedback_data).execute()
        
        return FeedbackResponse(**response.data[0])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback/problem/{pid}", response_model=List[FeedbackResponse], tags=["Feedback"])
async def get_problem_feedback(pid: int):
    """Get all feedback for a specific problem"""
    try:
        response = supabase.table("feedback").select("*").eq("pid", pid).execute()
        return [FeedbackResponse(**item) for item in response.data]
    except Exception as e:
        logger.error(f"Error fetching feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# NOTIFICATIONS
# =====================================================

@app.get("/notifications/mine", response_model=List[NotificationResponse], tags=["Notifications"])
async def get_my_notifications(
    unread_only: bool = Query(False),
    current_user = Depends(get_current_user)
):
    """Get notifications for current user"""
    try:
        query = supabase.table("notifications").select("*").eq("uid", current_user.uid)
        
        if unread_only:
            query = query.eq("read", False)
        
        response = query.order("created_at", desc=True).limit(50).execute()
        
        return [NotificationResponse(**item) for item in response.data]
        
    except Exception as e:
        logger.error(f"Error fetching notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/notifications/{notification_id}/read", tags=["Notifications"])
async def mark_notification_read(
    notification_id: int,
    current_user = Depends(get_current_user)
):
    """Mark notification as read"""
    try:
        supabase.table("notifications").update({"read": True}).eq("notification_id", notification_id).eq("uid", current_user.uid).execute()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/notifications/read-all", tags=["Notifications"])
async def mark_all_notifications_read(current_user = Depends(get_current_user)):
    """Mark all notifications as read"""
    try:
        supabase.table("notifications").update({"read": True}).eq("uid", current_user.uid).execute()
        return {"status": "success", "message": "All notifications marked as read"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# DEPARTMENTS & ISSUE TYPES
# =====================================================

@app.get("/departments", response_model=List[DepartmentResponse], tags=["Reference Data"])
async def get_departments():
    """Get all active departments"""
    try:
        response = supabase.table("departments").select("*").eq("active", True).execute()
        return [DepartmentResponse(**item) for item in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/issue-types", response_model=List[IssueTypeResponse], tags=["Reference Data"])
async def get_issue_types():
    """Get all active issue types"""
    try:
        response = supabase.table("issue_types").select("*").eq("active", True).execute()
        return [IssueTypeResponse(**item) for item in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# ANALYTICS & DASHBOARD
# =====================================================

@app.get("/analytics/dashboard", tags=["Analytics"])
async def get_dashboard_stats(
    department_id: Optional[int] = Query(None),
    current_user = Depends(get_department_staff)
):
    """
    Get dashboard statistics (Department staff/Admin only)
    
    Returns comprehensive analytics including:
    - Total issues
    - Resolution rates
    - Department performance
    - Issue type distribution
    """
    try:
        # Base query
        query = supabase.table("problems").select("*", count="exact")
        
        # Filter by department if provided
        if department_id:
            query = query.eq("department_id", department_id)
        elif current_user.department_id and current_user.role == "department_staff":
            query = query.eq("department_id", current_user.department_id)
        
        # Total issues
        all_response = query.execute()
        total_issues = all_response.count or 0
        
        # Resolved issues
        resolved_response = query.eq("status", "Resolved").execute()
        resolved_count = resolved_response.count or 0
        
        # Pending issues
        pending_response = query.in_("status", ["Reported", "Verified", "Assigned", "In Progress"]).execute()
        pending_count = pending_response.count or 0
        
        # Resolution rate
        resolution_rate = (resolved_count / total_issues * 100) if total_issues > 0 else 0
        
        # Average resolution time
        avg_time = 0
        if resolved_response.data:
            times = [p.get("actual_resolution_hours", 0) for p in resolved_response.data if p.get("actual_resolution_hours")]
            avg_time = sum(times) / len(times) if times else 0
        
        # High priority count
        high_priority_response = query.in_("priority", ["High", "Critical"]).execute()
        high_priority_count = high_priority_response.count or 0
        
        # Department stats
        dept_stats = []
        if not department_id:
            depts_response = supabase.table("departments").select("*").execute()
            for dept in depts_response.data:
                dept_problems = supabase.table("problems").select("*", count="exact").eq("department_id", dept["department_id"]).execute()
                dept_resolved = supabase.table("problems").select("*", count="exact").eq("department_id", dept["department_id"]).eq("status", "Resolved").execute()
                dept_in_progress = supabase.table("problems").select("*", count="exact").eq("department_id", dept["department_id"]).in_("status", ["Assigned", "In Progress"]).execute()
                
                total = dept_problems.count or 0
                resolved = dept_resolved.count or 0
                
                dept_stats.append({
                    "department_id": dept["department_id"],
                    "department_name": dept["name"],
                    "total_reported": total,
                    "total_resolved": resolved,
                    "in_progress": dept_in_progress.count or 0,
                    "avg_resolution_hours": dept.get("avg_resolution_hours", 72),
                    "resolution_rate": (resolved / total * 100) if total > 0 else 0
                })
        
        # Issue type distribution
        issue_type_stats = []
        types_response = supabase.table("issue_types").select("*").execute()
        for itype in types_response.data:
            type_problems = supabase.table("problems").select("*", count="exact").eq("issue_type_id", itype["issue_type_id"])
            if department_id:
                type_problems = type_problems.eq("department_id", department_id)
            type_response = type_problems.execute()
            count = type_response.count or 0
            
            if count > 0:
                issue_type_stats.append({
                    "issue_type_id": itype["issue_type_id"],
                    "issue_type_name": itype["name"],
                    "count": count,
                    "percentage": (count / total_issues * 100) if total_issues > 0 else 0
                })
        
        # Recent issues
        recent_response = supabase.table("problems").select("""
            *,
            issue_types(name),
            departments(name)
        """).order("reported_at", desc=True).limit(10).execute()
        
        recent_issues = []
        for item in recent_response.data:
            if item.get("issue_types"):
                item["issue_type_name"] = item["issue_types"]["name"]
            if item.get("departments"):
                item["department_name"] = item["departments"]["name"]
            recent_issues.append(item)
        
        return {
            "total_issues": total_issues,
            "resolved_issues": resolved_count,
            "pending_issues": pending_count,
            "resolution_rate": round(resolution_rate, 2),
            "avg_resolution_time": round(avg_time, 2),
            "high_priority_count": high_priority_count,
            "by_department": dept_stats,
            "by_issue_type": issue_type_stats,
            "recent_issues": recent_issues
        }
        
    except Exception as e:
        logger.error(f"Error generating dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# ADMIN ENDPOINTS
# =====================================================

@app.get("/admin/users", tags=["Admin"])
async def get_all_users(current_user = Depends(get_admin_user)):
    """Get all users (Admin only)"""
    try:
        response = supabase.table("users").select("uid, email, full_name, phone, role, department_id, created_at").execute()
        return {"status": "success", "count": len(response.data), "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/problems", tags=["Admin"])
async def get_all_problems_admin(
    status: Optional[str] = Query(None),
    department_id: Optional[int] = Query(None),
    priority: Optional[str] = Query(None),
    current_user = Depends(get_admin_user)
):
    """Get all problems with filters (Admin only)"""
    try:
        query = supabase.table("problems").select("""
            *,
            issue_types(name),
            departments(name),
            users!problems_uid_fkey(full_name, email)
        """)
        
        if status:
            query = query.eq("status", status)
        if department_id:
            query = query.eq("department_id", department_id)
        if priority:
            query = query.eq("priority", priority)
        
        response = query.order("reported_at", desc=True).execute()
        
        for item in response.data:
            if item.get("issue_types"):
                item["issue_type_name"] = item["issue_types"]["name"]
            if item.get("departments"):
                item["department_name"] = item["departments"]["name"]
            if item.get("users"):
                item["reporter_name"] = item["users"]["full_name"]
        
        return {"status": "success", "count": len(response.data), "data": response.data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# UTILITY ENDPOINTS
# =====================================================

@app.get("/image/{pid}", tags=["Utility"])
async def get_problem_image(pid: int):
    """Stream problem image"""
    try:
        response = supabase.table("problems").select("photo").eq("pid", pid).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Problem not found")
        
        image_url = response.data[0]["photo"]
        if not image_url:
            raise HTTPException(status_code=404, detail="No image available")
        
        r = requests.get(image_url, stream=True)
        return StreamingResponse(r.raw, media_type="image/jpeg")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.DEBUG)
app = app
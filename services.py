# services.py - Business Logic Services
import math
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from database import supabase
from config import settings
import logging

logger = logging.getLogger(__name__)

# =====================================================
# GEOSPATIAL SERVICES
# =====================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points
    
    Returns:
        float: Distance in kilometers
    """
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Overpass API mirrors for reliability
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

def get_osm_nearby_count(lat: float, lon: float, amenity_type: str, radius: int = None) -> int:
    """
    Fetch count of nearby OSM amenities using Overpass API
    
    Args:
        lat: Latitude
        lon: Longitude
        amenity_type: Type of amenity (hospital, school, etc.)
        radius: Search radius in meters
        
    Returns:
        int: Count of nearby amenities
    """
    if radius is None:
        radius = settings.OSM_SEARCH_RADIUS_M
    
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="{amenity_type}"](around:{radius},{lat},{lon});
      way["amenity"="{amenity_type}"](around:{radius},{lat},{lon});
      relation["amenity"="{amenity_type}"](around:{radius},{lat},{lon});
    );
    out;
    """
    
    for url in OVERPASS_URLS:
        try:
            response = requests.get(url, params={'data': overpass_query}, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            count = 0
            if "elements" in data:
                count = sum(1 for el in data["elements"] if el["type"] in ["node", "way", "relation"])
            return count
            
        except Exception as e:
            logger.warning(f"Overpass API mirror failed: {url}, error: {e}")
            continue
    
    logger.error("All Overpass API mirrors failed")
    return 0

# =====================================================
# SEVERITY & PRIORITY CALCULATION
# =====================================================

def calculate_severity_score(
    ml_result: Dict,
    latitude: float,
    longitude: float,
    issue_type_name: str
) -> Tuple[float, str]:
    """
    Calculate severity score based on ML results and location context
    
    Returns:
        tuple: (severity_score, overall_severity)
    """
    # ML-based severity weights
    severity_weights = {
        "minor_pothole": 1.0,
        "medium_pothole": 2.0,
        "major_pothole": 3.0,
        "broken_streetlight": 2.0,
        "water_leak": 2.5,
        "sewage_overflow": 3.5,
        "garbage_accumulation": 1.5,
        "unknown": 1.0
    }
    
    # Amenity weights (importance multiplier)
    amenity_weights = {
        "hospital": 3.0,
        "school": 2.0,
        "university": 2.0,
        "bank": 1.0,
        "police": 2.5,
        "fire_station": 3.0
    }
    
    # Calculate ML factor
    severity = ml_result.get("severity", "unknown")
    confidence = ml_result.get("confidence", 0.5)
    ml_weight = severity_weights.get(severity, 1.0)
    ml_factor = ml_weight * confidence
    
    # Calculate location context factor
    nearby_counts = {}
    try:
        for amenity in amenity_weights.keys():
            count = get_osm_nearby_count(latitude, longitude, amenity)
            nearby_counts[amenity] = count
    except Exception as e:
        logger.error(f"Error fetching amenities: {e}")
    
    amenity_factor = sum(
        nearby_counts.get(a, 0) * w 
        for a, w in amenity_weights.items()
    ) / 10.0
    
    # Final severity score
    final_score = ml_factor + amenity_factor
    
    # Classify overall severity
    if final_score < 1.5:
        overall_severity = "Low"
    elif final_score < 3.0:
        overall_severity = "Moderate"
    else:
        overall_severity = "High"
    
    return round(final_score, 2), overall_severity

def determine_priority(severity_score: float, issue_type_name: str) -> str:
    """
    Determine priority based on severity and issue type
    
    Returns:
        str: Priority level (Low, Medium, High, Critical)
    """
    # Critical issue types always get high priority
    critical_issues = [
        "Sewage Overflow",
        "Traffic Signal Malfunction",
        "Tree Fallen"
    ]
    
    if issue_type_name in critical_issues:
        return "Critical"
    
    # For other issues, use severity score
    if severity_score >= 3.5:
        return "Critical"
    elif severity_score >= 2.5:
        return "High"
    elif severity_score >= 1.5:
        return "Medium"
    else:
        return "Low"

# =====================================================
# DEPARTMENT ROUTING
# =====================================================

def auto_route_to_department(issue_type_id: int) -> Optional[int]:
    """
    Automatically route issue to appropriate department
    
    Returns:
        int: department_id
    """
    try:
        # Fetch issue type details
        response = supabase.table("issue_types").select("department_id").eq("issue_type_id", issue_type_id).execute()
        
        if response.data:
            return response.data[0].get("department_id")
        
        return None
    except Exception as e:
        logger.error(f"Error routing to department: {e}")
        return None

# =====================================================
# RESOLUTION TIME ESTIMATION
# =====================================================

def estimate_resolution_time(
    issue_type_id: int,
    priority: str,
    department_id: int
) -> Tuple[int, str]:
    """
    Estimate resolution time based on historical data and current workload
    
    Returns:
        tuple: (estimated_hours, confidence_level)
    """
    try:
        # Get department's average resolution time
        dept_response = supabase.table("departments").select("avg_resolution_hours").eq("department_id", department_id).execute()
        
        base_hours = 72  # Default 3 days
        if dept_response.data:
            base_hours = dept_response.data[0].get("avg_resolution_hours", 72)
        
        # Adjust based on priority
        priority_multipliers = {
            "Critical": 0.3,   # 30% of base time
            "High": 0.5,       # 50% of base time
            "Medium": 1.0,     # 100% of base time
            "Low": 1.5         # 150% of base time
        }
        
        multiplier = priority_multipliers.get(priority, 1.0)
        estimated_hours = int(base_hours * multiplier)
        
        # Check current workload (pending issues for this department)
        workload_response = supabase.table("problems").select("pid", count="exact").eq("department_id", department_id).in_("status", ["Reported", "Verified", "Assigned", "In Progress"]).execute()
        
        pending_count = workload_response.count if workload_response.count else 0
        
        # Adjust for workload
        if pending_count > 50:
            estimated_hours = int(estimated_hours * 1.5)
            confidence = "Low"
        elif pending_count > 20:
            estimated_hours = int(estimated_hours * 1.2)
            confidence = "Medium"
        else:
            confidence = "High"
        
        return estimated_hours, confidence
        
    except Exception as e:
        logger.error(f"Error estimating resolution time: {e}")
        return 72, "Low"

def calculate_actual_resolution_hours(reported_at: datetime, resolved_at: datetime) -> int:
    """Calculate actual time taken to resolve issue"""
    delta = resolved_at - reported_at
    return int(delta.total_seconds() / 3600)  # Convert to hours

# =====================================================
# STATUS MANAGEMENT
# =====================================================

def update_problem_status(
    pid: int,
    new_status: str,
    changed_by: int,
    notes: Optional[str] = None
) -> bool:
    """
    Update problem status and maintain history
    
    Returns:
        bool: Success status
    """
    try:
        # Get current status
        current_response = supabase.table("problems").select("status").eq("pid", pid).execute()
        
        if not current_response.data:
            logger.error(f"Problem {pid} not found")
            return False
        
        old_status = current_response.data[0].get("status")
        
        # Update problem status
        update_data = {"status": new_status, "updated_at": datetime.utcnow().isoformat()}
        
        # Add timestamp fields based on status
        if new_status == "Verified":
            update_data["verified_at"] = datetime.utcnow().isoformat()
        elif new_status == "Assigned":
            update_data["assigned_at"] = datetime.utcnow().isoformat()
        elif new_status == "Resolved":
            update_data["resolved_at"] = datetime.utcnow().isoformat()
            # Calculate actual resolution time
            problem = current_response.data[0]
            if problem.get("reported_at"):
                reported_at = datetime.fromisoformat(problem["reported_at"].replace('Z', '+00:00'))
                actual_hours = calculate_actual_resolution_hours(reported_at, datetime.utcnow())
                update_data["actual_resolution_hours"] = actual_hours
        elif new_status == "Closed":
            update_data["closed_at"] = datetime.utcnow().isoformat()
        
        supabase.table("problems").update(update_data).eq("pid", pid).execute()
        
        # Add to status history
        supabase.table("status_history").insert({
            "pid": pid,
            "old_status": old_status,
            "new_status": new_status,
            "changed_by": changed_by,
            "notes": notes
        }).execute()
        
        logger.info(f"Problem {pid} status updated: {old_status} -> {new_status}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating problem status: {e}")
        return False

# =====================================================
# ASSIGNMENT LOGIC
# =====================================================

def auto_assign_to_staff(department_id: int) -> Optional[int]:
    """
    Automatically assign issue to least loaded staff member
    
    Returns:
        int: User ID of assigned staff
    """
    try:
        # Get all active staff in this department
        staff_response = supabase.table("users").select("uid").eq("department_id", department_id).eq("role", "department_staff").execute()
        
        if not staff_response.data:
            logger.warning(f"No staff found for department {department_id}")
            return None
        
        staff_ids = [s["uid"] for s in staff_response.data]
        
        # Find staff with least assigned issues
        min_count = float('inf')
        assigned_staff = None
        
        for staff_id in staff_ids:
            count_response = supabase.table("problems").select("pid", count="exact").eq("assigned_to", staff_id).in_("status", ["Assigned", "In Progress"]).execute()
            
            count = count_response.count if count_response.count else 0
            
            if count < min_count:
                min_count = count
                assigned_staff = staff_id
        
        return assigned_staff
        
    except Exception as e:
        logger.error(f"Error auto-assigning to staff: {e}")
        return None

# =====================================================
# NOTIFICATION SERVICES
# =====================================================

def create_notification(
    uid: int,
    pid: int,
    notification_type: str,
    title: str,
    message: str
) -> bool:
    """Create notification for user"""
    try:
        supabase.table("notifications").insert({
            "uid": uid,
            "pid": pid,
            "type": notification_type,
            "title": title,
            "message": message,
            "sent_via": "app"
        }).execute()
        
        logger.info(f"Notification created for user {uid}")
        return True
    except Exception as e:
        logger.error(f"Error creating notification: {e}")
        return False

def notify_status_change(pid: int, new_status: str):
    """Send notification on status change"""
    try:
        # Get problem details
        problem = supabase.table("problems").select("uid, email").eq("pid", pid).execute()
        
        if not problem.data:
            return
        
        uid = problem.data[0]["uid"]
        
        # Create notification
        title = f"Issue #{pid} Status Updated"
        message = f"Your reported issue has been updated to: {new_status}"
        
        create_notification(uid, pid, "status_update", title, message)
        
    except Exception as e:
        logger.error(f"Error sending status notification: {e}")

def request_feedback(pid: int):
    """Request feedback after issue resolution"""
    try:
        problem = supabase.table("problems").select("uid").eq("pid", pid).execute()
        
        if not problem.data:
            return
        
        uid = problem.data[0]["uid"]
        
        title = f"Issue #{pid} Resolved - Feedback Requested"
        message = "Your issue has been resolved. Please share your feedback to help us improve."
        
        create_notification(uid, pid, "feedback_request", title, message)
        
    except Exception as e:
        logger.error(f"Error requesting feedback: {e}")

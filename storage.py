# storage.py - Supabase Storage Utilities
import uuid
from database import supabase
from config import settings
import logging

logger = logging.getLogger(__name__)

def upload_image_to_supabase(file_bytes: bytes, file_ext: str) -> str:
    """
    Upload image to Supabase Storage
    
    Args:
        file_bytes: Image bytes
        file_ext: File extension (jpg, png, etc.)
        
    Returns:
        str: Public URL of uploaded image
    """
    try:
        # Generate unique filename
        file_name = f"{uuid.uuid4()}.{file_ext}"
        
        # Upload to Supabase Storage
        supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).upload(
            file_name, 
            file_bytes
        )
        
        # Get public URL
        public_url = supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).get_public_url(file_name)
        
        logger.info(f"Image uploaded successfully: {file_name}")
        return public_url
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise Exception(f"Failed to upload image: {e}")

def delete_image_from_supabase(file_path: str) -> bool:
    """
    Delete image from Supabase Storage
    
    Args:
        file_path: Path/name of file in storage
        
    Returns:
        bool: Success status
    """
    try:
        supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).remove([file_path])
        logger.info(f"Image deleted: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return False

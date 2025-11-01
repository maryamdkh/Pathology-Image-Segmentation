import requests
from tqdm import tqdm
import os


def download_from_url(
    url: str, 
    save_path: str, 
    chunk_size: int = 1024
) -> bool:
    """
    Download a file from URL with progress bar and error handling.
    
    Args:
        url: The URL of the file to download
        save_path: Local path where the file should be saved
        chunk_size: Size of chunks to download at a time (in bytes)
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Start the download request
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Get total file size from headers
        total_size = int(response.headers.get('content-length', 0))
        
        # Initialize progress bar
        progress_bar = tqdm(
            total=total_size, 
            unit='iB', 
            unit_scale=True, 
            desc=f"Downloading {os.path.basename(save_path)}"
        )
        
        # Download and write file
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                progress_bar.update(len(chunk))
                file.write(chunk)
        
        progress_bar.close()
        
        # Verify download completeness
        if total_size != 0 and progress_bar.n != total_size:
            print("⚠️ Warning: Download incomplete.")
            return False
        else:
            print("✅ Download completed successfully!")
            print(f"📂 Saved to: {save_path}")
            return True
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        return False
    except IOError as e:
        print(f"❌ File operation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


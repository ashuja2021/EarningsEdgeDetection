import subprocess
import logging
import os
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dolt_installation():
    """Check if dolt is installed and accessible"""
    dolt_path = shutil.which('dolt')
    if dolt_path is None:
        logger.error("Dolt command not found in PATH. Please ensure Dolt is installed and added to system PATH")
        logger.error("Note: Dolt commands must be run from an administrator PowerShell window")
        return False
    logger.info(f"Found Dolt installation at: {dolt_path}")
    return True

def test_dolt_connection(database_path):
    logger.info(f"Testing connection to database at: {database_path}")
    
    if not os.path.exists(database_path):
        logger.error(f"Database path does not exist: {database_path}")
        return False
    
    # Simple query to test connection
    test_query = "SELECT 1 as test"
    
    try:
        result = subprocess.run(
            ["dolt", "sql", "-q", test_query, "--result-format", "csv"],
            cwd=database_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully connected to database at {database_path}")
            logger.info(f"Query result: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"Failed to connect to database at {database_path}")
            logger.error(f"Error: {result.stderr}")
            logger.error("Note: Dolt commands must be run from an administrator PowerShell window")
            return False
            
    except Exception as e:
        logger.error(f"Exception occurred while testing connection: {str(e)}")
        logger.error("Note: Dolt commands must be run from an administrator PowerShell window")
        return False

if __name__ == "__main__":
    # First check if dolt is installed
    if not check_dolt_installation():
        exit(1)
        
    # Test both databases
    options_db = "D:\\databases\\options"
    earnings_db = "D:\\databases\\earnings"
    
    logger.info("Testing Dolt database connections...")
    
    options_ok = test_dolt_connection(options_db)
    earnings_ok = test_dolt_connection(earnings_db)
    
    if options_ok and earnings_ok:
        logger.info("All database connections successful!")
    else:
        logger.error("One or more database connections failed!")
        logger.error("Please ensure you are running this script from an administrator PowerShell window") 
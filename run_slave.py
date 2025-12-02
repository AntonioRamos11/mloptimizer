import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from app.init_nodes import InitNodes

# Create logs directory structure if it doesn't exist
log_dir = Path('logs/slave/errors')
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
log_filename = log_dir / f'slave_errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function with exception handling"""
    try:
        logger.info("=" * 60)
        logger.info("Starting Slave Node")
        logger.info("=" * 60)
        
        slaveNode = InitNodes()
        logger.info("Slave node initialized successfully")
        
        slaveNode.slave()
        
    except KeyboardInterrupt:
        logger.info("\nSlave node stopped by user (Ctrl+C)")
        sys.exit(0)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("FATAL ERROR: Slave node crashed!")
        logger.error("=" * 60)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        logger.error(f"Error log saved to: {log_filename}")
        logger.error("=" * 60)
        sys.exit(1)

if __name__ == '__main__':
    main()
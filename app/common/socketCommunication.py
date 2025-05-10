from flask_socketio import SocketIO, send
import time
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MSGType():
    MASTER_STATUS = 'masterStatus'
    SLAVE_STATUS = 'slaveStatus'
    NEW_MODEL = 'newModel'
    MASTER_MODEL_COUNT = 'modelCount'
    FINISHED_MODEL = 'finishModel'
    CHANGE_PHASE = 'changePhase'
    MASTER_ERROR = 'masterError'
    FINISHED_TRAINING = 'finishedTrain'

class SocketCommunication:
    isSocket: bool = False
    socket: SocketIO
    
    @staticmethod    
    def decide_print_form(msgType: MSGType, info, max_retries=3, initial_backoff=1.0):
        """
        Send message with retry logic for RabbitMQ connection failures
        
        Args:
            msgType: Type of message to send
            info: Message payload
            max_retries: Maximum number of retry attempts (default: 3)
            initial_backoff: Initial backoff time in seconds (doubles each retry)
        """
        if not SocketCommunication.isSocket:
            print(info['msg'])
            return
            
        retry_count = 0
        backoff = initial_backoff
        
        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    logging.info(f"Retry attempt {retry_count}/{max_retries} for message type: {msgType}")
                
                # Attempt to send message via socket
                SocketCommunication.socket.emit(msgType, info)
                
                # If we get here, it was successful
                if retry_count > 0:
                    logging.info(f"Successfully sent message after {retry_count} retries")
                return
                
            except RuntimeError as e:
                # Specific handling for RabbitMQ connection closed error
                if "closed" in str(e):
                    logging.error(f"RabbitMQ connection closed: {e}")
                    error_details = {
                        'error': str(e),
                        'message_type': msgType,
                        'attempt': retry_count + 1
                    }
                    print(f"RabbitMQ connection error: {error_details}")
                else:
                    # Other runtime errors
                    logging.error(f"Runtime error: {e}")
                    
            except Exception as e:
                # Log any other exceptions
                logging.error(f"Error sending mess1age: {e}")
                logging.debug(traceback.format_exc())
            
            # If we've reached max retries, fallback to printing
            if retry_count >= max_retries:
                logging.warning(f"Max retries ({max_retries}) reached. Falling back to console output.")
                print(f"[SOCKET FALLBACK] {msgType}: {info['msg']}")
                return
                
            # Exponential backoff before next retry
            logging.info(f"Waiting {backoff:.2f}s before retry...")
            time.sleep(backoff)
            backoff *= 2  # Exponential backoff
            retry_count += 1
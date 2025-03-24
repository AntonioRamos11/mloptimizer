import os
import re
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def parse_debug_log(log_file="debug_strategy.log"):
    """Parse debug log to extract events and timing"""
    events = []
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found")
        return events
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                # Extract timestamp
                timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
                if timestamp_match:
                    timestamp = datetime.datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
                    
                    # Extract key events
                    if "Processing model response" in line:
                        model_id = re.search(r'id\': (\d+)', line)
                        if model_id:
                            events.append((timestamp, "model_response", f"Model {model_id.group(1)}"))
                    
                    elif "Building Hall of Fame" in line:
                        events.append((timestamp, "phase_change", "Hall of Fame Built"))
                    
                    elif "DEEP_TRAINING phase" in line:
                        events.append((timestamp, "phase_change", "Deep Training Started"))
                    
                    elif "Decided: GENERATE_MODEL" in line:
                        events.append((timestamp, "action", "Generate Model"))
                    
                    elif "Decided: WAIT" in line:
                        events.append((timestamp, "action", "Wait"))
                    
                    elif "ERROR" in line:
                        error_msg = re.search(r'ERROR.*?: (.*)', line)
                        if error_msg:
                            events.append((timestamp, "error", error_msg.group(1)[:20]))
            except Exception as e:
                print(f"Error parsing line: {line}\n{e}")
                
    return events

def visualize_optimization_flow(log_file="debug_strategy.log", output_file="optimization_flow.png"):
    """Create timeline visualization of events"""
    events = parse_debug_log(log_file)
    
    if not events:
        print("No events to visualize")
        return
    
    # Separate events by type
    timestamps = [e[0] for e in events]
    event_types = [e[1] for e in events]
    labels = [e[2] for e in events]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors for different event types
    colors = {
        'model_response': 'blue',
        'phase_change': 'green',
        'action': 'purple',
        'error': 'red'
    }
    
    # Plot events
    for i, (timestamp, event_type, label) in enumerate(events):
        color = colors.get(event_type, 'gray')
        ax.scatter(timestamp, i, c=color, s=100)
        ax.text(timestamp, i+0.1, label, fontsize=8, rotation=45, ha='right')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                      label=event_type.replace('_', ' ').title(),
                      markerfacecolor=color, markersize=10)
                      for event_type, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Format x-axis (time)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    
    # Set labels
    plt.title('Optimization Flow Timeline')
    plt.xlabel('Time')
    plt.ylabel('Event Sequence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
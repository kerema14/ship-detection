import subprocess

# Define the command as a list of strings
command = [
    "python", "validation.py",
    
    "--coco_path", "../coco",  
    "--model_path", "outputs/coco_retinanet_161.pt",
    
]

# Run the command using subprocess with real-time output
try:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Continuously read from stdout and stderr
    for line in process.stdout:
        print(line, end="")  # Print the output in real-time

    for line in process.stderr:
        print(line, end="")  # Print any errors in real-time

    # Wait for the process to finish
    process.wait()

except subprocess.CalledProcessError as e:
    print("Error running command:", e.stderr)

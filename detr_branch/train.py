import subprocess

# Define the command as a list of strings
command = [
    "python", "main.py",
    "--dataset_file", "custom",
    "--coco_path", "C:/Users/kerem/OneDrive/Masaüstü/Politechnika Warszawska/EARIN - Intro to Artificial Intelligence/ship-detection/detr_branch/custom/",
    "--output_dir", "outputs",
    "--resume", "detr-r50_no-class-head.pth",
    "--num_classes", "1",
    "--epochs", "300"
    "--lr_drop", "50",
    "--lr_drop_rate", "0.59234",
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

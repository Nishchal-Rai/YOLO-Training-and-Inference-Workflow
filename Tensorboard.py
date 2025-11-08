import os
import time
import webbrowser
from tensorboard import program

# -------------------------------
# 1. Paths to log folders
# -------------------------------
trained_base = r"C:\Users\raibi\PycharmProjects\Assessment3\TrainYourOwnYOLO\Data\Model_Weights\1762595530\train"
pretrained_base = r"C:\Users\raibi\PycharmProjects\Assessment3\TrainYourOwnYOLO\2_Training\Data\Model_Weights\pretrained_train"

# -------------------------------
# 2. Detect stage1/stage2 logs
# -------------------------------
def get_log_subfolders(base_folder):
    if not os.path.exists(base_folder):
        raise FileNotFoundError(f"Folder does not exist: {base_folder}")
    return [
        os.path.join(base_folder, f)
        for f in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, f)) and
        any(fname.startswith("events.out.tfevents") for fname in os.listdir(os.path.join(base_folder, f)))
    ] or [base_folder]

trained_logs = get_log_subfolders(trained_base)
pretrained_logs = get_log_subfolders(pretrained_base)

# -------------------------------
# 3. Create a parent logdir for TensorBoard
# -------------------------------
# TensorBoard now requires a single logdir or dictionary-like format
all_logs = trained_logs + pretrained_logs
print("Detected the following logs:")
for log in all_logs:
    print(log)

# -------------------------------
# 4. Launch TensorBoard with serve
# -------------------------------
tb = program.TensorBoard()
tb.configure(argv=[
    "serve",  # required subcommand for modern TensorBoard
    "--logdir", os.path.commonpath(all_logs),
    "--port", "6006",
    "--bind_all"
])

url = tb.launch()
webbrowser.open(url)
print(f"TensorBoard running at: {url}")

# -------------------------------
# 5. Keep script alive
# -------------------------------
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("TensorBoard stopped manually")

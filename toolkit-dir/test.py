import os

os.system('python evaluate_tracker.py --workspace_path ../workspace-dir --tracker simplified_mosse')
os.system('python compare_trackers.py --workspace_path ../workspace-dir --trackers simplified_mosse ncc_tracker --sensitivity 100')

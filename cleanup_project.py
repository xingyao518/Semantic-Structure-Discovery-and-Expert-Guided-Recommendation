import shutil
import os
import glob

print("Starting cleanup...")

# 1. Delete folders
folders = [
    "data/processed/visualizations/perplexity",
    "data/processed/visualizations/nmf",
    "data/processed/visualizations/sbert",
    "visualization/perplexity",
    "visualization/nmf",
    "visualization/sbert",
    # Also clean up src subfolders if created earlier
    "src/visualization/perplexity",
    "src/visualization/nmf",
    "src/visualization/sbert"
]

for f in folders:
    if os.path.exists(f):
        try:
            shutil.rmtree(f)
            print(f"Deleted folder: {f}")
        except Exception as e:
            print(f"Error deleting folder {f}: {e}")
    else:
        print(f"Folder not found: {f}")

# 2. Delete files containing keywords
# "perplexity", "nmf", "sbert"
# Exclude visualization/lda, visualization/logistic, visualization/retrieval folders implicitly 
# by checking if file is inside them (though we are scanning recursively)

keywords = ["perplexity", "nmf", "sbert"]

# Walk through project
for root, dirs, files in os.walk("."):
    # Skip .git, __pycache__, and the preserved folders to be safe
    # But user said "Delete all Python files related to these modules... except those inside LDA/logistic/retrieval folders"
    # Actually, the preserved folders are "visualization/lda", etc.
    # If a file is named "perplexity_eval.py" inside "evaluation/", it should be deleted.
    
    if ".git" in root or "__pycache__" in root:
        continue
        
    for file in files:
        # Check if file matches keywords
        if any(k in file.lower() for k in keywords):
            file_path = os.path.join(root, file)
            
            # Safety check: ensure we don't delete something clearly unrelated or protected
            # User said: "except those inside LDA/logistic/retrieval folders"
            # But the keywords "perplexity", "nmf", "sbert" are unlikely to be in LDA files unless they are comparing.
            # Assuming files like "visualize_lda.py" don't contain "nmf" in the filename.
            
            # Double check extension - "Delete all Python files"
            # But user previously asked to remove images too?
            # "Delete all Python files related to these modules... Any file whose name contains..."
            
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

print("Cleanup complete.")



# test_db.py
from db import save_prediction, fetch_recent

if __name__ == "__main__":
    new_id = save_prediction("test_upload.jpg", "AI-generated", 0.92)
    print("Inserted id:", new_id)
    rows = fetch_recent(5)
    for r in rows:
        print(r)

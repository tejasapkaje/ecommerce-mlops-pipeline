import schedule
import time
import subprocess
import os
import sys

# ==========================================
# PATH CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RETRAIN_SCRIPT = os.path.join(BASE_DIR, 'src', 'retrain_pipeline.py')

def run_retraining():
    print("\n" + "="*50)
    print("⏰ [AUTO-SCHEDULER] Time to get smarter! Starting automatic retraining...")
    print("="*50)
    
    try:
        # sys.executable ensures that it uses the Python from your Virtual Environment
        subprocess.run([sys.executable, RETRAIN_SCRIPT], check=True)
        print("\n✅ [AUTO-SCHEDULER] Automatic retraining finished successfully!")
    except Exception as e:
        print(f"\n❌ [AUTO-SCHEDULER] Error during automated retraining: {e}")

# ==========================================
# SET SCHEDULE AUTOMATION RULES HERE
# ==========================================

# Option A: Run every 1 minute (BEST FOR TESTING RIGHT NOW)
# schedule.every(1).minutes.do(run_retraining)

# Option B: Run every day at midnight (Best for Production)
# schedule.every().day.at("00:00").do(run_retraining)

# Option C: Run every Sunday at 2 AM
schedule.every().sunday.at("02:00").do(run_retraining)

print("🤖 Auto-Scheduler Bot is now running in the background...")
print("⏳ Waiting for the scheduled time. Press CTRL+C to stop this bot.")

# The Infinite Loop that keeps the bot alive
while True:
    schedule.run_pending()
    time.sleep(1) # Wait 1 second before checking the clock again
    

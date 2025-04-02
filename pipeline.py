import subprocess
from datetime import datetime
import time
import sys

def run(script_name):
    print(f"\n🔧 Running: {script_name}")
    start = time.time()

    try:
        subprocess.run([sys.executable, f"scripts/{script_name}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}: {e}")
        return

    duration = round(time.time() - start, 2)
    print(f"✅ {script_name} completed in {duration}s")

if __name__ == "__main__":
    print(f"\n🚀 Pipeline started @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    run("import_har_data.py")
    run("train_har_model.py")
    run("predict_har.py")
    run("analyze_har.py")

    print(f"\n✅ All steps completed @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

import subprocess
from datetime import datetime
import time

def run(script_name):
    print(f"\n🔧 Running: {script_name}")
    start = time.time()

    try:
        subprocess.run(["python", f"scripts/{script_name}"], check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Error running {script_name}")
        return

    duration = round(time.time() - start, 2)
    print(f"✅ {script_name} completed in {duration}s")

if __name__ == "__main__":
    print(f"\n🚀 Pipeline started @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    run("import_har_data.py")     # generate fresh fake logs
    run("analyse_har.py")   # extract features from latest logs
    run("train_har_model.py")      # train model on features
    run("predict_har.py")    # run predictions using latest model

    print(f"\n✅ All steps completed @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

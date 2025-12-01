import tkinter as tk
from tkinter import messagebox, filedialog
import subprocess
import os
import sys
import shutil

print("目前使用的 Python 路徑：", sys.executable)

# Java 路徑
JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-8.0.472.8-hotspot"

# 指定 Python 3.10 可執行檔
PYTHON_EXE = r"C:\Users\Leon\AppData\Local\Programs\Python\Python310\python.exe"
# Hadoop winutils 根目錄與執行檔
HADOOP_HOME = r"C:\Winutils"
WINUTILS_BIN = os.path.join(HADOOP_HOME, 'bin')
WINUTILS_EXE = os.path.join(WINUTILS_BIN, 'winutils.exe')
# Spark checkpoint 目錄
CHECKPOINT_DIR = r"C:\tmp\spark-checkpoint"

selected_script = None

def select_script():
    global selected_script
    path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")])
    if path:
        selected_script = path
        status_label.config(text=f"選擇腳本：{os.path.basename(path)}")

def run_pyspark():
    if not selected_script:
        messagebox.showwarning("提醒", "請先選擇 .py 腳本")
        return

    env = os.environ.copy()

    # 正確位置：這裡才設定 SPARK_HOME
    env['SPARK_HOME'] = r"C:\spark\spark-3.5.1-bin-hadoop3"
    env['PATH'] = r"C:\spark\spark-3.5.1-bin-hadoop3\bin;" + env['PATH']

    # Java
    env['JAVA_HOME'] = JAVA_HOME
    env['PATH'] = os.path.join(JAVA_HOME, 'bin') + os.pathsep + env['PATH']

    # winutils
    env['HADOOP_HOME'] = HADOOP_HOME
    env['PATH'] = WINUTILS_BIN + os.pathsep + env['PATH']

    # Python
    env['PYSPARK_PYTHON'] = PYTHON_EXE
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)

    # 執行
    subprocess.Popen([
        PYTHON_EXE,
        selected_script
    ], env=env, cwd=os.path.dirname(selected_script)).communicate()
    try:
        # 確保 winutils.exe 存在
        if not os.path.exists(WINUTILS_EXE):
            messagebox.showerror("錯誤", f"找不到 winutils.exe，請確認路徑：{WINUTILS_EXE}")
            return

        # 建立 checkpoint 目錄
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)

        # 執行 PySpark 腳本
        process = subprocess.Popen(
            [PYTHON_EXE, selected_script],
            env=env,
            cwd=os.path.dirname(selected_script),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate()

        if process.returncode == 0:
            messagebox.showinfo("成功", "PySpark 腳本執行完成！")
        else:
            messagebox.showerror("錯誤", f"執行失敗：\n{stderr}")

    except Exception as e:
        messagebox.showerror("錯誤", f"執行失敗：{e}")

# 建立 GUI
root = tk.Tk()
root.title('簡易 PySpark 啟動器')
root.geometry('500x200')

tk.Label(root, text='選擇要執行的 PySpark 腳本 (.py)').pack(pady=10)
select_btn = tk.Button(root, text='選擇腳本', command=select_script, bg='lightblue')
select_btn.pack()
status_label = tk.Label(root, text='尚未選擇腳本', fg='gray')
status_label.pack(pady=5)

run_btn = tk.Button(root, text='執行 PySpark', command=run_pyspark, bg='lightgreen')
run_btn.pack(pady=15)

root.mainloop()
